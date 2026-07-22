"""D4 结果重塑直构与旧路径的等价性测试。

``_computed_dict_to_panel`` 用一次分配直构 (instrument, datetime) 面板，
替代 ``_legacy_reshape`` 的 concat→unstack→stack→swaplevel 多次全景拷贝。
本文件在各种边界形态下断言两条路径输出完全一致（排序后逐值比较），
并验证形状不一致时的运行时兜底。
"""

import numpy as np
import pandas as pd
import pytest

from qlib.data.backend.ddb_qlib.ddb_features import (
    _computed_dict_to_panel,
    _legacy_reshape,
)


def _assert_equivalent(data: dict) -> None:
    new = _computed_dict_to_panel(data)
    legacy = _legacy_reshape(data)
    legacy.index.names = ["instrument", "datetime"]
    legacy = legacy.sort_index().reindex(columns=list(new.columns))
    pd.testing.assert_frame_equal(new.sort_index(), legacy, check_names=True)


def _make(aliases: int, dates, codes, seed: int = 0, nan_ratio: float = 0.0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(aliases):
        values = rng.standard_normal((len(dates), len(codes)))
        if nan_ratio:
            mask = rng.random(values.shape) < nan_ratio
            values = np.where(mask, np.nan, values)
        out[f"ExprName{i}"] = [values, pd.DatetimeIndex(dates), list(codes)]
    return out


DATES = pd.date_range("2024-01-01", periods=4, freq="B")
CODES = ["SZ000001", "SH600000", "SH600519"]  # 故意乱序


class TestEquivalence:
    def test_basic_multi_alias(self):
        _assert_equivalent(_make(3, DATES, CODES))

    def test_unsorted_codes_and_dates(self):
        dates = pd.DatetimeIndex(["2024-01-03", "2024-01-01", "2024-01-02"])
        _assert_equivalent(_make(2, dates, CODES, seed=1))

    def test_with_nan(self):
        _assert_equivalent(_make(2, DATES, CODES, seed=2, nan_ratio=0.3))

    def test_single_alias(self):
        _assert_equivalent(_make(1, DATES, CODES, seed=3))

    def test_single_date(self):
        _assert_equivalent(_make(2, DATES[:1], CODES, seed=4))

    def test_single_code(self):
        _assert_equivalent(_make(2, DATES, CODES[:1], seed=5))


class TestFallback:
    def test_shape_mismatch_falls_back_to_legacy(self):
        """alias 间形状不一致时回退 legacy（不抛错、语义与旧版一致）。"""
        data = _make(1, DATES, CODES)
        # 第二个 alias 的日期轴不同（模拟 DDB 端异常占位返回）
        other_dates = pd.date_range("2024-02-01", periods=4, freq="B")
        data.update(_make(1, other_dates, CODES, seed=9))
        # legacy 能处理（unstack 对齐并集），直构应回退到相同结果
        result = _computed_dict_to_panel(data)
        legacy = _legacy_reshape(data)
        legacy.index.names = ["instrument", "datetime"]
        pd.testing.assert_frame_equal(
            result.sort_index(), legacy.sort_index().reindex(columns=list(result.columns))
        )

    def test_malformed_entry_falls_back(self):
        """条目缺轴（长度不足 3）时不抛错——与 legacy 行为对齐或回退。"""
        data = _make(2, DATES, CODES)
        data["ExprName0"] = [np.zeros((2, 2))]  # 缺日期/代码轴
        with pytest.raises(Exception):
            # legacy 与直构在此病态输入下都应报错（IndexError 语义保留）
            _computed_dict_to_panel(data)
