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

from qlib.data.backend.ddb_qlib import invalidate_ddb_caches
from qlib.data.backend.ddb_qlib.ddb_features import fetch_features_from_ddb


@pytest.fixture(autouse=True)
def _clear_caches():
    """D1/D3 引入进程内缓存后，每个测试都从干净缓存开始以保证计数确定性。"""
    invalidate_ddb_caches()
    yield
    invalidate_ddb_caches()

# 3 个交易日 × 2 只股票的固定小样本
CAL = make_calendar("2024-01-01", 5)
DATES = pd.DatetimeIndex(CAL[:3])
CODES = ["SH600000", "SZ000001"]
START, END = "2024-01-01", "2024-01-03"

FEATURE_KEY = ("dfs://QlibFeaturesDay", "Features")


def _pure_table_result(script: str) -> pd.DataFrame:
    """按查询构造纯字段分支的长表结果（code/date/close/open）。"""
    rows = [
        {"code": c, "date": d, "close": float(i + 1), "open": float((i + 1) * 10)}
        for i, (c, d) in enumerate((c, d) for d in DATES for c in CODES)
    ]
    return pd.DataFrame(rows)


def _make_pure_session() -> RecordingSession:
    # D2 后纯字段分支为单条 SQL 脚本（select ... from loadTable(...)）
    return RecordingSession(
        calendar=CAL,
        table_columns={FEATURE_KEY: ["code", "date", "close", "open", "volume"]},
        run_responses=[(r"select .*from loadTable", _pure_table_result)],
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
        # 单脚本查询：code in instruments 过滤 + 日期字面量 + 排序
        script = session.run_scripts[-1]
        assert "code in instruments" in script
        assert "date between pair(2024.01.01,2024.01.03)" in script
        assert "order by date, code" in script
        # 上传的变量含 instruments（且仅 instruments——纯字段分支不再上传表达式）
        assert session.uploads[-1] == {"instruments": CODES}

    def test_dict_instruments_uses_conditional_filter(self):
        session = _make_pure_session()
        df = fetch_features_from_ddb(session, SPANS, ["$close"], START, END, "day")

        assert not df.empty
        # spans 路径：映射创建与 conditionalFilter 主查询合并在同一脚本（单次往返）
        script = session.run_scripts[-1]
        assert "createDateStockMapping(2024.01.01,2024.01.03,instruments)" in script
        assert "conditionalFilter(code, date, codeRangeFilter)" in script
        assert session.counts["run"] == 1

    def test_missing_base_field_padded_with_zero(self):
        """表中不存在的基础字段用 '0 as 字段' 兜底。"""
        session = RecordingSession(
            calendar=CAL,
            table_columns={FEATURE_KEY: ["code", "date", "close"]},
            run_responses=[
                (
                    r"select .*from loadTable",
                    pd.DataFrame(
                        {"code": CODES, "date": [DATES[0]] * 2, "close": [1.0, 2.0], "not_exist": [0, 0]}
                    ),
                )
            ],
        )
        fetch_features_from_ddb(session, CODES, ["$close", "$not_exist"], START, END, "day")
        assert "0 as not_exist" in session.run_scripts[-1]


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

    def test_unrecognized_token_triggers_alpha_retry(self):
        """D5 兜底：未识别函数错误时全量加载 alpha 库并重试一次。"""
        calls = {"n": 0}

        def _fe(script):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("Syntax Error: Cannot recognize the token gtjaAlpha3")
            return _fe_response(script)

        session = RecordingSession(
            calendar=CAL, run_responses=[(r"FeatureEngineeringByDate", _fe)]
        )
        df = fetch_features_from_ddb(session, CODES, self.FIELDS, START, END, "day")
        assert not df.empty
        loaded = {str(f).rsplit("/", 1)[-1] for f in session.run_files}
        assert {"gtja191Alpha.dos", "qlib158Alpha.dos", "wq101alpha.dos"} <= loaded
        assert calls["n"] == 2

    def test_non_token_error_does_not_retry(self):
        """非未识别函数错误：不重试、保留 RuntimeError 语义与异常链。"""

        def _fe(script):
            raise RuntimeError("Server response: out of memory")

        session = RecordingSession(
            calendar=CAL, run_responses=[(r"FeatureEngineeringByDate", _fe)]
        )
        with pytest.raises(RuntimeError, match="DolphinDB 因子计算失败"):
            fetch_features_from_ddb(session, CODES, self.FIELDS, START, END, "day")
        assert session.run_files == []  # 未触发兜底加载

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

    D1（日历缓存）+ D2（往返缩减）后的目标状态：
    计算分支每批 = 1 upload + 1 run（日历预热后）；
    纯字段分支 = 1 upload + 1 run + schema 查询（D3 缓存后消失）。
    """

    def test_computed_branch_rpc_counts(self):
        session = _make_computed_session()
        fetch_features_from_ddb(session, CODES, ["Ref($close,1)"], START, END, "day")
        assert session.counts["loadTable"] == 1  # 仅日历首次下载（跨调用共享）
        assert session.counts["run"] == 1  # 仅主查询（日期已内联为字面量）
        assert session.counts["upload"] == 1

    def test_pure_list_branch_rpc_counts(self):
        session = _make_pure_session()
        fetch_features_from_ddb(session, CODES, ["$close"], START, END, "day")
        # 首次调用：日历下载 + build_field_expr schema 共 2 次 loadTable
        assert session.counts["loadTable"] == 2
        assert session.counts["run"] == 1  # 单脚本主查询
        assert session.counts["upload"] == 1
        # 二次调用：日历与 schema 均命中缓存，仅 1 upload + 1 run
        fetch_features_from_ddb(session, CODES, ["$close"], START, END, "day")
        assert session.counts["loadTable"] == 2, "D3 回归：schema/日历缓存未生效"
        assert session.counts["run"] == 2
        assert session.counts["upload"] == 2

    def test_calendar_downloaded_once_across_calls(self):
        """D1 回归：日历经模块级缓存共享，跨调用只下载一次（曾经每次都下载）。"""
        session = _make_computed_session()
        fetch_features_from_ddb(session, CODES, ["Ref($close,1)"], START, END, "day")
        fetch_features_from_ddb(session, CODES, ["Ref($close,1)"], START, END, "day")
        calendar_loads = [c for c in session.load_table_calls if c[0] == "dfs://QlibCalendars"]
        assert len(calendar_loads) == 1, "日历缓存失效：跨调用重复全量下载"
