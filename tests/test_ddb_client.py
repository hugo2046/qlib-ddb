"""DDBClient 连接池生命周期的离线单元测试（不依赖 DolphinDB 服务器）。

历史 bug 回归：
1. ``_pool_instance`` 曾是类变量——连接不同服务器的多个客户端会共享同一个池；
2. ``close_pool`` 曾引用不存在的 ``cls._pool_lock``（AttributeError 被裸 except 吞掉），
   且读取的类变量永远是 None，实际从未关闭过任何池；
3. ``tableAppender``/``tableUpsert`` 调用不存在的 ``self.get_session()``，属死代码，已删除。
"""

import pytest

from qlib.data.backend.ddb_qlib import ddb_client as ddb_client_mod
from qlib.data.backend.ddb_qlib.ddb_client import DDBClient, DDBConnectionSpec


class _FakeSession:
    """记录 connect/close 调用的假会话。"""

    def __init__(self, *args, **kwargs):
        self.connect_calls = []
        self.closed = False

    def connect(self, **kwargs):
        self.connect_calls.append(kwargs)

    def close(self):
        self.closed = True


class _FakePool:
    """记录构造参数与 shutDown 调用的假连接池。"""

    instances: list["_FakePool"] = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.shutdown_called = 0
        _FakePool.instances.append(self)

    def shutDown(self):
        self.shutdown_called += 1


@pytest.fixture()
def fake_ddb(monkeypatch):
    """把 ddb_client 模块内的 dolphindb SDK 替换为假实现。"""
    _FakePool.instances = []
    monkeypatch.setattr(ddb_client_mod.ddb, "session", _FakeSession)
    monkeypatch.setattr(ddb_client_mod.ddb, "DBConnectionPool", _FakePool)

    class _FakeSettings:
        PROTOCOL_DDB = "ddb"

    monkeypatch.setattr(ddb_client_mod.ddb, "settings", _FakeSettings, raising=False)
    return _FakePool


def _make_client(uri: str = "dolphindb://admin:pwd@127.0.0.1:8848") -> DDBClient:
    return DDBClient(DDBConnectionSpec(uri=uri))


def test_pool_created_lazily_and_once(fake_ddb):
    client = _make_client()
    assert fake_ddb.instances == [], "连接池不应在构造时创建"
    pool_first = client.pool
    pool_second = client.pool
    assert pool_first is pool_second
    assert len(fake_ddb.instances) == 1, "连接池应只创建一次"


def test_pool_not_shared_between_clients(fake_ddb):
    """回归：池曾是类变量，导致不同客户端共享同一个池。"""
    client_a = _make_client("dolphindb://admin:pwd@10.0.0.1:8848")
    client_b = _make_client("dolphindb://admin:pwd@10.0.0.2:8848")
    assert client_a.pool is not client_b.pool
    assert len(fake_ddb.instances) == 2


def test_close_pool_shuts_down_and_resets(fake_ddb):
    client = _make_client()
    pool = client.pool
    client.close_pool()
    assert pool.shutdown_called == 1
    # 关闭后再次访问会新建
    assert client.pool is not pool
    # 未创建时关闭是空操作，不会新建池
    created_before = len(fake_ddb.instances)
    fresh = _make_client()
    fresh.close_pool()
    assert len(fake_ddb.instances) == created_before


def test_close_pool_swallows_shutdown_error(fake_ddb):
    client = _make_client()
    pool = client.pool
    pool.shutDown = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    client.close_pool()  # 不应抛出
    assert client._pool_instance is None


def test_close_closes_session_and_pool(fake_ddb):
    client = _make_client()
    pool = client.pool
    client.close()
    assert pool.shutdown_called == 1
    assert client.session.closed is True


def test_provider_pool_is_lazy(fake_ddb, monkeypatch):
    """回归：DolphinDBClientProvider 曾在 init 时急切创建 4 连接的池（无人使用）。"""
    import qlib.data.backend.ddb_qlib as ddb_pkg
    from qlib.data.data import DolphinDBClientProvider

    monkeypatch.setattr(ddb_pkg, "register_ddb_functions_to_qlib", lambda session: None)
    provider = DolphinDBClientProvider(uri="dolphindb://admin:pwd@127.0.0.1:8848")
    assert fake_ddb.instances == [], "provider 初始化不应创建连接池"
    _ = provider.pool
    assert len(fake_ddb.instances) == 1, "首次访问 pool 才创建连接池"


def test_get_shared_client_reuses_per_uri(fake_ddb):
    """D6：同 URI 复用共享客户端，不同 URI 各自独立。"""
    from qlib.data.backend.ddb_qlib import ddb_operator as op

    op._shared_clients.clear()
    a = op.get_shared_client("dolphindb://admin:pwd@10.0.0.1:8848")
    b = op.get_shared_client("dolphindb://admin:pwd@10.0.0.1:8848")
    c = op.get_shared_client("dolphindb://admin:pwd@10.0.0.2:8848")
    assert a is b
    assert a is not c
    op._shared_clients.clear()


def test_write_df_to_ddb_uses_shared_client(fake_ddb, monkeypatch):
    """D6 回归：write_df_to_ddb 曾每次调用新建 DDBClient（新 TCP 会话）。"""
    import pandas as pd

    from qlib.data.backend.ddb_qlib import ddb_operator as op

    op._shared_clients.clear()
    clients_used = []

    class _FakeOperator:
        def __init__(self, client):
            clients_used.append(client)

        def table_appender(self, **kwargs):
            pass

        def table_upsert(self, **kwargs):
            pass

    monkeypatch.setattr(op, "DDBTableOperator", _FakeOperator)
    uri = "dolphindb://admin:pwd@10.0.0.3:8848"
    op.write_df_to_ddb("db", "tb", pd.DataFrame({"a": [1]}), uri=uri)
    op.write_df_to_ddb("db", "tb", pd.DataFrame({"a": [1]}), uri=uri)
    assert len(clients_used) == 2
    assert clients_used[0] is clients_used[1], "同 URI 的两次写入应复用同一客户端"
    # 显式传 client 时优先使用
    explicit = _make_client("dolphindb://admin:pwd@10.0.0.4:8848")
    op.write_df_to_ddb("db", "tb", pd.DataFrame({"a": [1]}), uri=None, client=explicit)
    assert clients_used[-1] is explicit
    op._shared_clients.clear()


def test_dead_write_methods_removed(fake_ddb):
    """守卫：死代码 tableAppender/tableUpsert 不应被重新引入（正确实现在 DDBTableOperator）。"""
    client = _make_client()
    assert not hasattr(client, "tableAppender")
    assert not hasattr(client, "tableUpsert")
    assert not hasattr(client, "get_session")
