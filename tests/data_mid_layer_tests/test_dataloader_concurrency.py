# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""QlibDataLoader.load() 并发回归测试。

背景
----
QlibDataLoader.load() 并发调用时会发生跨线程数据污染：线程 A 请求
instruments X、线程 B 请求 instruments Y（X∩Y=∅），并发执行后 A 的返回里
混入 Y 的 instruments。根因：load() → load_group_df() → qlib 表达式求值会
读写 qlib 的全局数据层（DolphinDB session / 内部缓存），该层非线程安全，
并发求值时共享缓冲被互相覆盖。

该 race 仅在 DolphinDB 共享 session 下复现（文件后端按 instrument 分别读
独立文件，无共享结果缓冲，实测不污染）。因此本测试不依赖任何真实后端，
而是注入一个 ``load_group_df`` 子类，用一个「共享可变缓冲」精确模拟 DDB
非线程安全层的互相覆盖行为，从而在本机即可确定性复现红、验证锁修复绿。

被测对象始终是真实的 ``QlibDataLoader.load``（继承自 DLWParser）；子类只
替换数据 fetch 路径，锁的串行化契约在真实 ``load`` 入口生效。
"""

import threading
import time
from typing import List

import pandas as pd
import pytest

from qlib.data.dataset.loader import QlibDataLoader


class _UnsafeSharedLayer:
    """模拟 DolphinDB 非线程安全的全局数据层（共享 session / 缓存）。

    多个 load_group_df 调用共用同一个 ``buffer``：写入请求的 instruments，
    在「计算」期间可被其他并发线程覆盖——这正是 DDB session 复用导致的污染源。
    """

    def __init__(self) -> None:
        self.buffer: List[str] | None = None

    def query(self, instruments: List[str]) -> List[str]:
        """模拟一次 DDB 查询：写入 buffer → 计算期间可被覆盖 → 返回 buffer。

        返回的可能是「别的线程最后写入的 instruments」，污染由此体现。
        不清空 buffer，保证并发交错时返回值始终是 instruments 列表（要么是
        自己的、要么被对方覆盖），使污染通过断言而非异常暴露。
        """
        self.buffer = list(instruments)
        # 拉长临界区，扩大 race 窗口，使并发交错近乎必然
        time.sleep(0.05)
        return self.buffer  # ⚠️ 此处可能拿到对方线程覆盖后的 instruments


def _make_loader_cls(shared: _UnsafeSharedLayer) -> type:
    """构造一个 QlibDataLoader 子类：load_group_df 走模拟的 DDB 非安全层。"""

    class _ConcurrentQlibDataLoader(QlibDataLoader):
        # 与 init_qlib_once 的 _init_lock 同范式：类级锁，跨所有实例共享
        # （实际修复在 QlibDataLoader.load 中加锁，此处仅用于回归验证）

        def load_group_df(
            self,
            instruments,
            exprs: list,
            names: list,
            start_time=None,
            end_time=None,
            gp_name: str = None,
        ) -> pd.DataFrame:
            leaked = shared.query(list(instruments))
            idx = pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2020-01-01"), i) for i in leaked],
                names=["datetime", "instrument"],
            )
            return pd.DataFrame({names[0]: 0.0}, index=idx)

    return _ConcurrentQlibDataLoader


def _returned_instruments(df: pd.DataFrame) -> set:
    """从 load 返回的 DataFrame 中取出 instrument 集合。"""
    if df.index.nlevels > 1:
        return set(df.index.get_level_values("instrument"))
    return set(df.index)


# 两组不相交的 instruments
SZ = ["SZ000001", "SZ000002", "SZ000008", "SZ000009", "SZ000012", "SZ000021", "SZ000027", "SZ000029"]
SH = ["SH600000", "SH600004", "SH600006", "SH600007", "SH600008", "SH600009", "SH600010", "SH600011"]
assert not (set(SZ) & set(SH))


def _run_concurrently_and_assert(config) -> None:
    """并发对两个独立 loader 调 load()，断言返回集合各自 == 请求集合、不混入对方。"""
    shared = _UnsafeSharedLayer()
    LoaderCls = _make_loader_cls(shared)
    loader_sz = LoaderCls(config=config, freq="day")
    loader_sh = LoaderCls(config=config, freq="day")

    out: dict = {}
    barrier = threading.Barrier(2)

    def _call(loader, instruments, key):
        barrier.wait()  # 尽量同时进入 load()
        out[key] = loader.load(instruments=instruments, start_time="2020-01-01", end_time="2020-01-02")

    t_sz = threading.Thread(target=_call, args=(loader_sz, SZ, "sz"))
    t_sh = threading.Thread(target=_call, args=(loader_sh, SH, "sh"))
    t_sz.start()
    t_sh.start()
    t_sz.join()
    t_sh.join()

    sz_got = _returned_instruments(out["sz"])
    sh_got = _returned_instruments(out["sh"])
    # 各自的返回必须严格等于请求集合，不混入对方的任何 instrument
    assert sz_got == set(SZ), f"SZ loader 被污染：多余 instruments={sorted(sz_got - set(SZ))}"
    assert sh_got == set(SH), f"SH loader 被污染：多余 instruments={sorted(sh_got - set(SH))}"


@pytest.mark.parametrize("round_idx", list(range(40)))
def test_load_concurrent_no_cross_contamination_non_group(round_idx) -> None:
    """非 group 分支：并发 load 不应跨线程串数据。"""
    config = (["Mean($close, 5)", "$volume"], ["mean5", "vol"])
    _run_concurrently_and_assert(config)


@pytest.mark.parametrize("round_idx", list(range(40)))
def test_load_concurrent_no_cross_contamination_group(round_idx) -> None:
    """is_group=True 分支：锁同样必须覆盖 group 字典的 load 路径。"""
    config = {
        "feature": (["Mean($close, 5)"], ["mean5"]),
        "label": (["$volume"], ["vol"]),
    }
    _run_concurrently_and_assert(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
