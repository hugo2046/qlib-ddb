"""DDB 会话级锁的并发回归测试（离线，不依赖 DolphinDB 服务器）。

背景：ddb.Session 非线程安全，且 feature 查询是「run(上传日期)→upload(变量)→run(主查询)」
的多步会话对话；并发线程交错执行会互相覆盖服务器端的 instruments/expressions 变量，
造成跨线程数据污染（A 线程拿到 B 线程的股票池）。

约定（B1）：所有 ``DBClient.session`` 触点必须持有 ``DBClient.session_lock``（RLock）。
本测试验证 ``DBFeatureStorage.__getitem__`` 对整个 fetch 会话对话持锁串行化。
"""

import threading
import time

import pandas as pd
import pytest

import qlib.data.backend.ddb_qlib as ddb_pkg
from qlib.data.data import DBClient
from qlib.data.storage.dolphindb_storage import DBFeatureStorage


class _FakeProvider:
    """带真实 RLock 的假 DolphinDBClientProvider。"""

    def __init__(self):
        self.session = object()  # 会话本体不被本测试触碰
        self.session_lock = threading.RLock()


@pytest.fixture()
def fake_provider():
    old = DBClient.__dict__.get("_provider")
    provider = _FakeProvider()
    DBClient.register(provider)
    yield provider
    DBClient.register(old)


def test_feature_fetch_serialized_across_threads(fake_provider, monkeypatch):
    """并发调用 DBFeatureStorage 时，fetch 会话对话必须整体串行（不被交错）。"""
    events: list[tuple[int, str]] = []
    events_guard = threading.Lock()

    def fake_fetch(session, instruments, fields, start_time, end_time, freq):
        tid = threading.get_ident()
        with events_guard:
            events.append((tid, "enter"))
        time.sleep(0.02)  # 放大交错窗口
        with events_guard:
            events.append((tid, "exit"))
        return pd.DataFrame({"close": [1.0]})

    monkeypatch.setattr(ddb_pkg, "fetch_features_from_ddb", fake_fetch)

    storage = DBFeatureStorage(instrument=["SH600000"], field="$close", freq="day")

    threads = [threading.Thread(target=lambda: storage[:]) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(events) == 8
    # enter/exit 必须严格成对且属于同一线程——任何交错都意味着锁失效
    for i in range(0, len(events), 2):
        assert events[i][1] == "enter" and events[i + 1][1] == "exit", f"事件交错: {events}"
        assert events[i][0] == events[i + 1][0], f"跨线程交错: {events}"


def test_session_lock_is_reentrant(fake_provider):
    """session_lock 必须可重入（RLock）：storage 读取可嵌套在 feature 查询持锁期间。"""
    lock = DBClient.session_lock
    assert lock.acquire(blocking=False)
    try:
        assert lock.acquire(blocking=False), "session_lock 必须是 RLock（可重入）"
        lock.release()
    finally:
        lock.release()


def test_load_lock_scoped_to_ddb_backend(monkeypatch):
    """B2 回归：全局 _load_lock 仅在 DolphinDB 后端持有；文件后端 load 无锁。"""
    import qlib.data.data as data_mod
    from qlib.data.dataset.loader import DLWParser, QlibDataLoader

    lock_states: list[bool] = []

    def fake_super_load(self, instruments=None, start_time=None, end_time=None):
        lock_states.append(QlibDataLoader._load_lock.locked())
        return pd.DataFrame()

    monkeypatch.setattr(DLWParser, "load", fake_super_load)
    loader = object.__new__(QlibDataLoader)  # 绕过构造器，load 不依赖实例属性

    monkeypatch.setattr(data_mod, "is_using_dolphindb", lambda: True)
    loader.load()
    monkeypatch.setattr(data_mod, "is_using_dolphindb", lambda: False)
    loader.load()

    assert lock_states == [True, False], "DDB 路径应持锁，文件路径应无锁"


def test_provider_exposes_session_lock():
    """真实 DolphinDBClientProvider 必须暴露 session_lock（接口契约）。"""
    import inspect

    from qlib.data.data import DolphinDBClientProvider

    source = inspect.getsource(DolphinDBClientProvider.__init__)
    assert "session_lock" in source and "RLock" in source
