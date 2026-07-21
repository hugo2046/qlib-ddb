"""fetch_features_from_ddb 四大分支的离线回归测试（D0 安全网）。

在任何性能重构（往返缩减/缓存/重塑重写）之前固化当前行为：
- 纯字段 × list / dict(spans) 两分支
- 计算表达式 × list / dict(spans) 两分支
- 空 instruments 与 DDB 空 dict 的早退路径
- RPC 往返数基线（后续 D1/D2 优化后按目标值更新）

使用 tests/ddb_mocks.py 的 RecordingSession，不依赖 DolphinDB 服务器。
"""

import numpy as np
import pandas as pd
import pytest

from ddb_mocks import FakeQueryChain, RecordingSession, make_calendar

from qlib.data.backend.ddb_qlib.ddb_features import TradeDateUtils, fetch_features_from_ddb


@pytest.fixture(autouse=True)
def _clear_calendar_cache():
    """D1 引入模块级日历缓存后，每个测试都从干净缓存开始以保证计数确定性。"""
    TradeDateUtils.clear_cache()
    yield
    TradeDateUtils.clear_cache()

# 3 个交易日 × 2 只股票的固定小样本
CAL = make_calendar("2024-01-01", 5)
DATES = pd.DatetimeIndex(CAL[:3])
CODES = ["SH600000", "SZ000001"]
START, END = "2024-01-01", "2024-01-03"

FEATURE_KEY = ("dfs://QlibFeaturesDay", "Features")


def _pure_table_result(chain: FakeQueryChain) -> pd.DataFrame:
    """按查询构造纯字段分支的长表结果（code/date/close/open）。"""
    rows = [
        {"code": c, "date": d, "close": float(i + 1), "open": float((i + 1) * 10)}
        for i, (c, d) in enumerate((c, d) for d in DATES for c in CODES)
    ]
    return pd.DataFrame(rows)


def _make_pure_session() -> RecordingSession:
    return RecordingSession(
        calendar=CAL,
        table_columns={FEATURE_KEY: ["code", "date", "close", "open", "volume"]},
        table_results={FEATURE_KEY: _pure_table_result},
    )


def _fe_response(script: str) -> dict:
    """计算分支：模拟 FeatureEngineeringByDate 返回 {alias: [矩阵, 日期, 代码]}。"""
    values0 = np.arange(6, dtype=float).reshape(3, 2)  # dates × codes
    values1 = values0 * 100
    return {
        "ExprName0": [values0, DATES, CODES],
        "ExprName1": [values1, DATES, CODES],
    }


def _make_computed_session() -> RecordingSession:
    return RecordingSession(
        calendar=CAL,
        run_responses=[(r"FeatureEngineeringByDate", _fe_response)],
    )


SPANS = {
    "SH600000": [(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"))],
    "SZ000001": [(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03"))],
}


class TestPureFieldsBranch:
    def test_list_instruments(self):
        session = _make_pure_session()
        df = fetch_features_from_ddb(session, CODES, ["$close", "$open"], START, END, "day")

        assert list(df.columns) == ["$close", "$open"]
        assert df.index.names == ["instrument", "datetime"]
        assert len(df) == 6
        # 查询链：code in instruments 过滤 + date between + 排序
        chain = session.query_chains[-1]
        assert any("code in instruments" in w for w in chain.wheres)
        assert any("date between pair(2024.01.01,2024.01.03)" in w for w in chain.wheres)
        assert chain.sorts == [["date", "code"]]
        # 上传的变量含 instruments
        assert session.uploads[-1]["instruments"] == CODES

    def test_dict_instruments_uses_conditional_filter(self):
        session = _make_pure_session()
        df = fetch_features_from_ddb(session, SPANS, ["$close"], START, END, "day")

        assert not df.empty
        # spans 路径：先建日期-股票映射，再用 conditionalFilter 过滤
        assert any("createDateStockMapping" in s for s in session.run_scripts)
        chain = session.query_chains[-1]
        assert any("conditionalFilter(code, date, codeRangeFilter)" in w for w in chain.wheres)

    def test_missing_base_field_padded_with_zero(self):
        """表中不存在的基础字段用 '0 as 字段' 兜底。"""
        session = RecordingSession(
            calendar=CAL,
            table_columns={FEATURE_KEY: ["code", "date", "close"]},
            table_results={
                FEATURE_KEY: lambda chain: pd.DataFrame(
                    {"code": CODES, "date": [DATES[0]] * 2, "close": [1.0, 2.0], "not_exist": [0, 0]}
                )
            },
        )
        fetch_features_from_ddb(session, CODES, ["$close", "$not_exist"], START, END, "day")
        chain = session.query_chains[-1]
        assert ["code", "date", "close", "0 as not_exist"] in chain.selects


class TestComputedBranch:
    FIELDS = ["Ref($close,1)", "Mean($open,5)"]

    def test_list_instruments(self):
        session = _make_computed_session()
        df = fetch_features_from_ddb(session, CODES, self.FIELDS, START, END, "day")

        # 列名映射回原始表达式（空格移除后）
        assert list(df.columns) == ["Ref($close,1)", "Mean($open,5)"]
        assert df.index.names == ["instrument", "datetime"]
        # 全组合：2 codes × 3 dates
        assert len(df) == 6
        assert df.index.is_monotonic_increasing
        # 数值正确性：ExprName0 矩阵为 dates×codes 的 0..5
        assert df.loc[("SH600000", DATES[0]), "Ref($close,1)"] == 0.0
        assert df.loc[("SZ000001", DATES[2]), "Ref($close,1)"] == 5.0
        assert df.loc[("SZ000001", DATES[2]), "Mean($open,5)"] == 500.0
        # 脚本用 instruments 变量（list 分支）
        fe_script = [s for s in session.run_scripts if "FeatureEngineeringByDate" in s][0]
        assert "FeatureEngineeringByDate(instruments," in fe_script

    def test_dict_instruments_applies_spans_mask(self):
        session = _make_computed_session()
        df = fetch_features_from_ddb(session, SPANS, self.FIELDS, START, END, "day")

        # dict 分支：脚本传键列表，spans 掩码在 Python 侧补齐
        fe_script = [s for s in session.run_scripts if "FeatureEngineeringByDate" in s][0]
        assert "FeatureEngineeringByDate(keys(instruments)," in fe_script
        # SH600000 的 2024-01-03 已出池，应被掩码剔除
        assert ("SH600000", DATES[2]) not in df.index
        assert ("SZ000001", DATES[2]) in df.index
        assert len(df) == 5

    def test_empty_result_dict_returns_empty(self):
        session = RecordingSession(
            calendar=CAL,
            run_responses=[(r"FeatureEngineeringByDate", {})],
        )
        df = fetch_features_from_ddb(session, CODES, self.FIELDS, START, END, "day")
        assert df.empty


class TestEarlyReturns:
    def test_empty_instruments_returns_empty(self):
        session = _make_computed_session()
        df = fetch_features_from_ddb(session, [], ["Ref($close,1)"], START, END, "day")
        assert df.empty
        # 早退不应触发上传与主查询
        assert session.counts["upload"] == 0


class TestRpcCountBaseline:
    """RPC 往返数基线（优化后按新目标更新本测试）。

    当前实现每次调用：日历全量下载(loadTable) + run(上传日期) +
    upload(变量) + 主查询。D1（日历缓存）与 D2（往返缩减）落地后应下降。
    """

    def test_computed_branch_rpc_counts(self):
        session = _make_computed_session()
        fetch_features_from_ddb(session, CODES, ["Ref($close,1)"], START, END, "day")
        assert session.counts["loadTable"] == 1  # 日历下载（D1 目标: 跨调用共享后为 0）
        assert session.counts["run"] == 2  # 上传日期 + 主查询（D2 目标: 1）
        assert session.counts["upload"] == 1

    def test_pure_list_branch_rpc_counts(self):
        session = _make_pure_session()
        fetch_features_from_ddb(session, CODES, ["$close"], START, END, "day")
        # 日历 + build_field_expr schema + 主查询共 3 次 loadTable
        assert session.counts["loadTable"] == 3
        assert session.counts["run"] == 1  # 上传日期
        assert session.counts["upload"] == 1

    def test_calendar_downloaded_once_across_calls(self):
        """D1 回归：日历经模块级缓存共享，跨调用只下载一次（曾经每次都下载）。"""
        session = _make_computed_session()
        fetch_features_from_ddb(session, CODES, ["Ref($close,1)"], START, END, "day")
        fetch_features_from_ddb(session, CODES, ["Ref($close,1)"], START, END, "day")
        calendar_loads = [c for c in session.load_table_calls if c[0] == "dfs://QlibCalendars"]
        assert len(calendar_loads) == 1, "日历缓存失效：跨调用重复全量下载"
