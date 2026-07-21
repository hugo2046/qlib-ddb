"""ddb_dataset_processor 的离线回归测试（不依赖 DolphinDB 服务器）。

历史 bug 回归：当传入 inst_processors 时，`ddb_dataset_processor` 曾在并行处理后
提前 `return pd.DataFrame()`，把处理结果整体丢弃并跳过 cache_to_origin_data 归一化。
本文件通过注入假的 ExpressionD provider 与顺序执行的 ParallelExt 存根，
在无 DolphinDB 连接的情况下覆盖该路径。
"""

import numpy as np
import pandas as pd
import pytest

from qlib.data import data as data_mod
from qlib.data.data import DatasetProvider
from qlib.data.inst_processor import InstProcessor


class _AddConst(InstProcessor):
    """给所有列加常数的标记处理器，用于验证处理结果没有被丢弃。"""

    def __init__(self, const: float = 100.0):
        self.const = const

    def __call__(self, df: pd.DataFrame, instrument, *args, **kwargs):
        return df + self.const


class _FakeExpressionProvider:
    """按固定值构造 (instrument, datetime) MultiIndex 结果的假 provider。

    :param instruments: 股票代码列表
    :param dates: 交易日列表
    """

    def __init__(self, instruments: list[str], dates: pd.DatetimeIndex):
        self.instruments = instruments
        self.dates = dates
        self.calls: list[list[str]] = []  # 记录每次调用的字段批次

    def expression(self, inst, fields, start_time, end_time, freq):
        self.calls.append(list(fields))
        index = pd.MultiIndex.from_product(
            [self.instruments, self.dates], names=["instrument", "datetime"]
        )
        # 每个字段填充其在全部调用中的序号，便于断言列取值
        data = {f: float(i) for i, f in enumerate(fields)}
        return pd.DataFrame(data, index=index, dtype=np.float32)


class _SequentialParallel:
    """顺序执行 joblib delayed 任务的 ParallelExt 存根。"""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, tasks):
        return [func(*args, **kwargs) for func, args, kwargs in tasks]


@pytest.fixture()
def fake_env(monkeypatch):
    """注入假 provider / 并行存根 / 内核数配置，返回 provider 供断言。"""
    instruments = ["SH600000", "SZ000001"]
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    provider = _FakeExpressionProvider(instruments, dates)
    monkeypatch.setattr(data_mod.ExpressionD, "_provider", provider)
    monkeypatch.setattr(data_mod, "ParallelExt", _SequentialParallel)
    monkeypatch.setattr(type(data_mod.C), "get_kernels", lambda self, freq: 1, raising=False)
    return provider


def test_inst_processors_result_not_discarded(fake_env):
    """回归：inst_processors 非空时结果必须保留（历史上被丢弃返回空 DataFrame）。"""
    result = DatasetProvider.ddb_dataset_processor(
        inst=["SH600000", "SZ000001"],
        column_names=["$close", "$open"],
        start_time=pd.Timestamp("2024-01-01"),
        end_time=pd.Timestamp("2024-01-03"),
        freq="day",
        inst_processors=[_AddConst(100.0)],
    )
    assert not result.empty, "inst_processors 处理结果被丢弃（回归到历史 bug）"
    # 基础值为字段序号（0/1），处理器加 100
    assert (result["$close"] == 100.0).all()
    assert (result["$open"] == 101.0).all()
    # 归一化路径必须走到：列名与索引结构保持约定
    assert list(result.columns) == ["$close", "$open"]
    assert result.index.names == ["instrument", "datetime"]
    # 结果按索引排序（确定性保证）
    assert result.index.is_monotonic_increasing


def test_without_inst_processors_unchanged(fake_env):
    """无 inst_processors 时行为与原实现一致。"""
    result = DatasetProvider.ddb_dataset_processor(
        inst=["SH600000", "SZ000001"],
        column_names=["$close"],
        start_time=pd.Timestamp("2024-01-01"),
        end_time=pd.Timestamp("2024-01-03"),
        freq="day",
    )
    assert not result.empty
    assert list(result.columns) == ["$close"]
    assert (result["$close"] == 0.0).all()


def test_column_chunking_over_30(fake_env):
    """超过 30 个字段按 30 一批分块调用 ExpressionD.expression。"""
    fields = [f"$f{i}" for i in range(65)]
    result = DatasetProvider.ddb_dataset_processor(
        inst=["SH600000", "SZ000001"],
        column_names=fields,
        start_time=pd.Timestamp("2024-01-01"),
        end_time=pd.Timestamp("2024-01-03"),
        freq="day",
    )
    assert [len(c) for c in fake_env.calls] == [30, 30, 5]
    assert list(result.columns) == fields


def test_inst_processors_with_chunking(fake_env):
    """分块 + inst_processors 组合路径：列齐全且处理生效。"""
    fields = [f"$f{i}" for i in range(31)]
    result = DatasetProvider.ddb_dataset_processor(
        inst=["SH600000", "SZ000001"],
        column_names=fields,
        start_time=pd.Timestamp("2024-01-01"),
        end_time=pd.Timestamp("2024-01-03"),
        freq="day",
        inst_processors=[_AddConst(1000.0)],
    )
    assert not result.empty
    assert list(result.columns) == fields
    # 第二批只有 1 个字段，序号从 0 重新计：$f30 基础值为 0
    assert (result["$f30"] == 1000.0).all()
