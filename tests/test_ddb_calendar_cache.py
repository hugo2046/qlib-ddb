"""D1 日历缓存的离线回归测试。

历史问题：
1. ``TradeDateUtils.__init__`` 每次构造都全量下载交易日历——而
   ``fetch_features_from_ddb`` 每批字段（30 个/批）都会构造一次，
   Alpha158 一次 ``D.features`` ≈ 6 次全量日历下载；
2. ``DBCalendarStorage.index()``/``__getitem__`` 绕过 ``H["c"]`` 缓存
   直接 ``_read_calendar()``，数据集对齐期间反复全量下载。

修复：TradeDateUtils 模块级缓存 + storage 统一走 ``_cached_calendar()``；
写路径通过 ``invalidate_ddb_caches()`` 失效。
"""

import threading

import numpy as np
import pandas as pd
import pytest

from ddb_mocks import RecordingSession, make_calendar

from qlib.data.backend.ddb_qlib import invalidate_ddb_caches
from qlib.data.backend.ddb_qlib.ddb_features import TradeDateUtils
from qlib.data.data import DBClient

CAL = make_calendar("2024-01-01", 10)


@pytest.fixture(autouse=True)
def _clean_caches():
    invalidate_ddb_caches()
    yield
    invalidate_ddb_caches()


class TestTradeDateUtilsCache:
    def test_calendar_loaded_once_across_instances(self):
        session = RecordingSession(calendar=CAL)
        utils_a = TradeDateUtils(session, "day")
        utils_b = TradeDateUtils(session, "day")
        assert session.counts["loadTable"] == 1, "第二次构造应命中缓存"
        # 缓存共享同一份数据且行为一致
        assert np.array_equal(utils_a._calendar, utils_b._calendar)
        start, end = utils_b.get_locate_date("2024-01-06", "2024-01-08")
        assert pd.Timestamp(start) == pd.Timestamp("2024-01-08")  # 周六周日 snap 到下一交易日

    def test_clear_cache_forces_reload(self):
        session = RecordingSession(calendar=CAL)
        TradeDateUtils(session, "day")
        TradeDateUtils.clear_cache()
        TradeDateUtils(session, "day")
        assert session.counts["loadTable"] == 2

    def test_invalidate_ddb_caches_clears_calendar(self):
        session = RecordingSession(calendar=CAL)
        TradeDateUtils(session, "day")
        invalidate_ddb_caches()
        TradeDateUtils(session, "day")
        assert session.counts["loadTable"] == 2


class _FakeProvider:
    def __init__(self, session):
        self.session = session
        self.session_lock = threading.RLock()


@pytest.fixture()
def calendar_storage(monkeypatch):
    """构造带假 provider 的 DBCalendarStorage。"""
    from qlib.config import C
    from qlib.data.storage.dolphindb_storage import DBCalendarStorage

    if "region" not in C:
        C["region"] = "cn"  # DBCalendarStorage.__init__ 依赖，qlib.init 时才会设置

    session = RecordingSession(
        table_results={
            ("dfs://QlibCalendars", "day"): pd.DataFrame({"TRADE_DAYS": pd.DatetimeIndex(CAL)})
        }
    )
    old = DBClient.__dict__.get("_provider")
    DBClient.register(_FakeProvider(session))
    storage = DBCalendarStorage(freq="day", future=False)
    yield storage, session
    DBClient.register(old)


class TestDBCalendarStorageCache:
    def test_index_and_getitem_share_cache(self, calendar_storage):
        storage, session = calendar_storage
        _ = storage.data
        loads_after_data = session.counts["loadTable"]
        _ = storage[0]
        _ = storage[1:3]
        loads_after_getitem = session.counts["loadTable"]
        assert loads_after_getitem == loads_after_data, (
            "index/__getitem__ 绕过缓存重复下载日历（历史回归）"
        )

    def test_data_values_unchanged_by_cache(self, calendar_storage):
        storage, session = calendar_storage
        assert list(storage.data) == list(pd.DatetimeIndex(CAL))
        assert storage[0] == pd.DatetimeIndex(CAL)[0]
