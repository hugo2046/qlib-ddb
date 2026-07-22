"""滚动算子取前序期（回看窗口外扩）的离线回归测试。

- 回看解析优先复用 qlib 算子树 ``get_extended_window_size``（嵌套/双臂/
  未来引用均覆盖）；qlib 无法实例化的表达式退回「正则扫窗口 + 配置兜底」。
- 计算分支脚本以 ``daysStep,lookbackDays,rightDays`` 尾参把外扩量传给
  DDB 端 ``FeatureEngineeringByDate``，服务器端外扩查询并截断回请求区间。

使用 tests/ddb_mocks.py 的 RecordingSession，不依赖 DolphinDB 服务器。
"""

import numpy as np
import pandas as pd
import pytest

from ddb_mocks import RecordingSession, make_calendar

from qlib.data.backend.ddb_qlib import invalidate_ddb_caches
from qlib.data.backend.ddb_qlib.ddb_features import (
    batch_extended_window,
    fetch_features_from_ddb,
    get_expression_extended_window,
)


@pytest.fixture(autouse=True)
def _clear_caches():
    invalidate_ddb_caches()
    yield
    invalidate_ddb_caches()


class TestExpressionExtendedWindow:
    """单表达式回看解析：qlib 算子树路径。"""

    @pytest.mark.parametrize(
        "expr,expected",
        [
            ("$close", (0, 0)),  # 纯字段无外扩
            ("Mean($close,20)", (19, 0)),  # 滚动窗口 N -> N-1
            ("Ref($close,5)", (5, 0)),  # 引用 N 期前 -> N
            ("Mean(Ref($close,5),20)", (24, 0)),  # 嵌套：5 + 19
            ("Corr($close,$volume,10)", (9, 0)),  # 双臂算子取 max
            ("Ref($close,-2)", (0, 2)),  # 未来引用 -> 向后外扩
            ("Ref($close,-2)/Ref($close,-1)-1", (0, 2)),  # 标签表达式
            ("Mean($close,20)/Std($close,60)", (59, 0)),  # 同表达式内取 max
        ],
    )
    def test_qlib_op_tree(self, expr, expected):
        assert get_expression_extended_window(expr, 252) == expected

    def test_fallback_regex_window(self):
        """qlib 不认识的函数：独立整数当窗口。"""
        assert get_expression_extended_window("myCustomOp($close,60)", 252) == (60, 0)

    def test_fallback_ignores_identifier_digits(self):
        """标识符里的数字（gtjaAlpha191_001）不当窗口，走配置兜底。"""
        assert get_expression_extended_window("gtjaAlpha191_001($close)", 100) == (100, 0)

    def test_fallback_future_reference(self):
        """兜底路径的未来引用：负数窗口仅按函数参数形式识别。"""
        lft, rght = get_expression_extended_window("myCustomOp($close, -3)", 252)
        assert rght == 3

    def test_fallback_ignores_large_scaling_constant(self):
        """大数值常量（缩放因子）不当窗口，避免误判为百万日回看，退回配置兜底。"""
        assert get_expression_extended_window(
            "gtjaAlpha191_005($volume/1000000, $close)", 252
        ) == (252, 0)

    def test_fallback_window_upper_bound(self):
        """上界内整数仍视为窗口；超过上界视为常量并退回兜底。"""
        assert get_expression_extended_window("myCustomOp($close, 2000)", 30)[0] == 2000
        assert get_expression_extended_window("myCustomOp($close, 2001)", 30)[0] == 30

    def test_fallback_future_ignores_large_constant(self):
        """未来引用兜底同样忽略超大常量，避免向后外扩到不合理天数。"""
        _, rght = get_expression_extended_window("myCustomOp($close, -9999999)", 252)
        assert rght == 0


class TestBatchExtendedWindow:
    def test_batch_takes_max(self):
        exprs = ["Mean($close,20)", "Ref($close,-2)", "Std($close,60)"]
        assert batch_extended_window(exprs, 252) == (59, 2)

    def test_empty_batch(self):
        assert batch_extended_window([], 252) == (0, 0)


# 3 个交易日 × 2 只股票的固定小样本（与 test_fetch_features.py 一致）
CAL = make_calendar("2024-01-01", 5)
DATES = pd.DatetimeIndex(CAL[:3])
CODES = ["SH600000", "SZ000001"]
START, END = "2024-01-01", "2024-01-03"


def _fe_response(script: str) -> dict:
    values = np.arange(6, dtype=float).reshape(3, 2)  # dates × codes
    return {"ExprName0": [values, DATES, CODES]}


def _make_session() -> RecordingSession:
    return RecordingSession(
        calendar=CAL,
        run_responses=[(r"FeatureEngineeringByDate", _fe_response)],
    )


class TestLookbackScriptWiring:
    """计算分支脚本携带 lookbackDays/rightDays 尾参。"""

    def test_rolling_lookback_in_script(self):
        session = _make_session()
        fetch_features_from_ddb(session, CODES, ["Mean($close,20)"], START, END, "day")
        fe_script = [s for s in session.run_scripts if "FeatureEngineeringByDate" in s][0]
        assert ",252,19,0)" in fe_script

    def test_label_right_extension_in_script(self):
        session = _make_session()
        fetch_features_from_ddb(
            session, CODES, ["Ref($close,-2)/Ref($close,-1)-1"], START, END, "day"
        )
        fe_script = [s for s in session.run_scripts if "FeatureEngineeringByDate" in s][0]
        assert ",252,0,2)" in fe_script

    def test_lookback_default_configurable(self, monkeypatch):
        """qlib 解析不了且扫不到窗口的表达式，用 C["ddb_lookback_default"] 兜底。"""
        from qlib.config import C

        monkeypatch.setitem(C, "ddb_lookback_default", 30)
        session = _make_session()
        fetch_features_from_ddb(
            session, CODES, ["gtjaAlphaCustom($close)"], START, END, "day"
        )
        fe_script = [s for s in session.run_scripts if "FeatureEngineeringByDate" in s][0]
        assert ",252,30,0)" in fe_script
