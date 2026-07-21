"""D3 存储层/schema/表达式翻译缓存的离线回归测试。

历史问题：
1. 每个存储访问器都 ``check()`` → ``existsTable`` RPC，无缓存；
2. ``DBInstrumentStorage`` 每次访问全量下载股票池表并行循环重建 dict；
3. ``build_field_expr``/``table_appender``/``table_upsert`` 每次调用重新
   拉取表 schema；
4. ``adapt_qlib_expr_syntax_for_ddb``（~100 行递归解析）每字段每次重跑。

修复：existsTable 仅正向缓存、股票池走 ``H["i"]``、schema 走
``get_table_columns`` 进程内缓存、表达式翻译走 lru_cache；
前三者由 ``invalidate_ddb_caches()`` 统一失效。
"""

import threading

import pandas as pd
import pytest

from ddb_mocks import RecordingSession

from qlib.data.backend.ddb_qlib import invalidate_ddb_caches
from qlib.data.backend.ddb_qlib.ddb_features import (
    OPERATOR_MAPPING,
    _adapt_cached,
    adapt_qlib_expr_syntax_for_ddb,
)
from qlib.data.backend.ddb_qlib.utils import get_table_columns
from qlib.data.data import DBClient


@pytest.fixture(autouse=True)
def _clean_caches():
    invalidate_ddb_caches()
    yield
    invalidate_ddb_caches()


class _FakeProvider:
    def __init__(self, session):
        self.session = session
        self.session_lock = threading.RLock()


@pytest.fixture()
def fake_provider():
    session = RecordingSession(
        table_results={
            ("dfs://QlibInstruments", "csi300"): pd.DataFrame(
                {
                    "instrument": ["SH600000", "SH600000", "SZ000001"],
                    "start_datetime": pd.to_datetime(["2020-01-01", "2022-01-01", "2020-01-01"]),
                    "end_datetime": pd.to_datetime(["2021-01-01", "2099-12-31", "2099-12-31"]),
                }
            )
        }
    )
    old = DBClient.__dict__.get("_provider")
    DBClient.register(_FakeProvider(session))
    yield session
    DBClient.register(old)


class TestExistsCache:
    def test_positive_result_cached(self, fake_provider):
        from qlib.data.storage.dolphindb_storage import DBFeatureStorage

        storage = DBFeatureStorage(instrument="SH600000", field="close", freq="day")
        storage.check()
        storage.check()
        assert fake_provider.counts["existsTable"] == 1, "正向 exists 结果应被缓存"

    def test_negative_result_not_cached(self, fake_provider):
        """表不存在必须持续报错（负结果不缓存），表创建后立即可见。"""
        from qlib.data.storage.dolphindb_storage import DBFeatureStorage

        fake_provider.exists_result = False
        storage = DBFeatureStorage(instrument="SH600000", field="close", freq="day")
        with pytest.raises(ValueError):
            storage.check()
        with pytest.raises(ValueError):
            storage.check()
        assert fake_provider.counts["existsTable"] == 2, "负结果不得缓存"
        # 表被创建后（exists 变 True）无需失效缓存即可通过
        fake_provider.exists_result = True
        storage.check()


class TestInstrumentCache:
    def test_read_once_across_accessors(self, fake_provider):
        from qlib.data.storage.dolphindb_storage import DBInstrumentStorage

        storage = DBInstrumentStorage(market="csi300", freq="day")
        data_a = storage.data
        data_b = storage.data
        assert len(storage) == 2
        assert fake_provider.counts["loadTable"] == 1, "股票池应经 H['i'] 缓存"
        # 内容正确且跨调用一致
        assert data_a == data_b
        assert data_a["SH600000"] == [
            (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")),
            (pd.Timestamp("2022-01-01"), pd.Timestamp("2099-12-31")),
        ]
        # 返回浅拷贝：调用方修改不污染缓存
        data_a["FAKE"] = []
        assert "FAKE" not in storage.data

    def test_invalidate_forces_reload(self, fake_provider):
        from qlib.data.storage.dolphindb_storage import DBInstrumentStorage

        storage = DBInstrumentStorage(market="csi300", freq="day")
        _ = storage.data
        invalidate_ddb_caches()
        _ = storage.data
        assert fake_provider.counts["loadTable"] == 2


class TestSchemaCache:
    def test_columns_loaded_once(self):
        session = RecordingSession(table_columns={("dfs://Db", "Tb"): ["code", "date", "close"]})
        cols_a = get_table_columns(session, "dfs://Db", "Tb")
        cols_b = get_table_columns(session, "dfs://Db", "Tb")
        assert cols_a == cols_b == ["code", "date", "close"]
        assert session.counts["loadTable"] == 1
        # 返回副本：调用方修改不污染缓存
        cols_a.append("hacked")
        assert get_table_columns(session, "dfs://Db", "Tb") == ["code", "date", "close"]

    def test_invalidate_forces_reload(self):
        session = RecordingSession(table_columns={("dfs://Db", "Tb"): ["code"]})
        get_table_columns(session, "dfs://Db", "Tb")
        invalidate_ddb_caches()
        get_table_columns(session, "dfs://Db", "Tb")
        assert session.counts["loadTable"] == 2


class TestExpressionMemoization:
    def test_cache_hit_and_identical_output(self):
        _adapt_cached.cache_clear()
        exprs = ["Ref($close,1)/$close-1", "Mean($volume,5)", "Std($close,20)"]
        first = [adapt_qlib_expr_syntax_for_ddb(e, OPERATOR_MAPPING, True) for e in exprs]
        second = [adapt_qlib_expr_syntax_for_ddb(e, OPERATOR_MAPPING, True) for e in exprs]
        assert first == second
        assert _adapt_cached.cache_info().hits >= len(exprs)

    def test_custom_mapping_bypasses_cache(self):
        """自定义映射（非默认 dict）不得命中默认映射的缓存。"""
        custom = dict(OPERATOR_MAPPING)
        custom["Ref"] = "customMove"
        result = adapt_qlib_expr_syntax_for_ddb("Ref($close,1)", custom)
        assert "customMove(" in result
        # 默认映射结果不受影响
        assert "move(" in adapt_qlib_expr_syntax_for_ddb("Ref($close,1)")
