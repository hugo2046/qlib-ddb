# tests/test_ddb_storage_spans.py
"""测试 DBFeatureStorage 对成分股 spans（入池/出池区间）的保持。

回归背景（HANDOFF_成分股spans过滤失效_20260720）：
    ``D.features(D.instruments("csi300"), ...)`` 在 DolphinDB 模式下，
    ``get_instruments_d`` 产出 ``{code: [(入池, 出池), ...]}`` 的 dict 并一路
    透传到 ``DBFeatureStorage.__init__``；但该构造函数原实现为::

        self.instrument = (
            instrument.upper()
            if isinstance(instrument, str)
            else list(map(lambda x: x.upper(), instrument))
        )

    dict 落入 else 分支，``map`` 迭代 dict 只取键——spans 被静默丢弃、退化为
    list。下游 ``fetch_features_from_ddb`` 收到 list 走 ``code in instruments``
    分支，无入池/出池日期过滤，导致返回历史所有曾入池股票的全区间行情
    （出池不截断、调离不剔除）。

    DDB 端本身对 dict 是完好支持的（纯字段分支 ``createDateStockMapping`` +
    ``conditionalFilter``；非纯字段分支 ``FeatureEngine.buildWhereConditions``），
    所以修复只需在存储层保持 dict 原型（仅键做 ``.upper()``）。

    本测试为离线纯单元测试，锁定 ``DBFeatureStorage.__init__`` 三种入参
    （str / list / dict）的规范化行为，不依赖 DolphinDB 服务器。
"""

import pandas as pd

from qlib.data.backend.ddb_qlib.ddb_features import apply_spans_mask
from qlib.data.storage.dolphindb_storage import DBFeatureStorage


def _make_storage(instrument):
    """构造 DBFeatureStorage（__init__ 无 DDB 连接副作用，可离线构造）。"""
    return DBFeatureStorage(instrument=instrument, field="close", freq="day")


class TestDBFeatureStorageInstrumentNormalization:
    """``DBFeatureStorage.__init__`` 对 instrument 入参的规范化。"""

    def test_dict_preserves_spans(self) -> None:
        """dict 入参必须保持 dict 原型：键 upper，spans 值原样保留。

        这是成分股动态过滤的关键：spans 丢失即退化为“历史全体成员全历史”。
        """
        spans = {
            "000008.sz": [
                (pd.Timestamp("2015-06-15"), pd.Timestamp("2018-06-08")),
            ],
            "000069.sz": [
                (pd.Timestamp("2016-12-12"), pd.Timestamp("2024-06-14")),
            ],
        }
        storage = _make_storage(spans)

        assert isinstance(storage.instrument, dict), "dict 入参不得退化为 list（spans 丢失）"
        assert set(storage.instrument) == {"000008.SZ", "000069.SZ"}
        assert storage.instrument["000008.SZ"] == spans["000008.sz"]
        assert storage.instrument["000069.SZ"] == spans["000069.sz"]

    def test_dict_multi_spans_preserved(self) -> None:
        """多段入池/出池区间（调出后再调入）逐段保留、顺序不变。"""
        spans = {
            "600000.sh": [
                (pd.Timestamp("2010-01-01"), pd.Timestamp("2015-05-29")),
                (pd.Timestamp("2019-12-16"), pd.Timestamp("2026-07-17")),
            ],
        }
        storage = _make_storage(spans)

        assert storage.instrument == {"600000.SH": spans["600000.sh"]}

    def test_str_upper_unchanged(self) -> None:
        """单标的 str 路径（LocalFeatureProvider.feature）行为保持：upper 后原样。"""
        storage = _make_storage("000300.sh")

        assert storage.instrument == "000300.SH"

    def test_list_upper_unchanged(self) -> None:
        """list 路径行为保持：逐元素 upper。"""
        storage = _make_storage(["000001.sz", "600000.sh"])

        assert storage.instrument == ["000001.SZ", "600000.SH"]


def _make_panel(codes_dates):
    """按 (code, [dates]) 列表构造 (instrument, datetime) MultiIndex 面板。"""
    frames = []
    for code, dates in codes_dates:
        idx = pd.MultiIndex.from_product(
            [[code], pd.to_datetime(dates)], names=["instrument", "datetime"]
        )
        frames.append(pd.DataFrame({"factor": range(len(dates))}, index=idx))
    return pd.concat(frames).sort_index()


class TestApplySpansMask:
    """``apply_spans_mask``：非纯字段分支的 Python 侧 spans 兜底掩码。

    背景：``FeatureEngineeringByDate`` 经 ``mr`` 分布式执行时 worker 端无法
    还原 ``conditionalFilter`` 所需的 dict，故非纯字段分支按键列表全量计算，
    再在 Python 侧按 spans 掩码——语义与原版 qlib ``inst_calculator``
    （先算后掩码，data.py:711-716）一致。
    """

    def test_rows_outside_spans_removed(self) -> None:
        """出池日之后的行被剔除，入池区间内的行保留。"""
        data = _make_panel(
            [
                ("000069.SZ", ["2024-06-13", "2024-06-14", "2024-06-17"]),
                ("600000.SH", ["2024-06-13", "2024-06-14", "2024-06-17"]),
            ]
        )
        spans = {
            "000069.SZ": [(pd.Timestamp("2016-12-12"), pd.Timestamp("2024-06-14"))],
            "600000.SH": [(pd.Timestamp("2010-01-01"), pd.Timestamp("2040-12-31"))],
        }
        result = apply_spans_mask(data, spans)

        d69 = result.xs("000069.SZ", level="instrument").index
        assert d69.max() <= pd.Timestamp("2024-06-14"), "出池后应截断"
        assert len(result.xs("600000.SH", level="instrument")) == 3, "在池内的行不应误删"

    def test_code_not_in_spans_removed_entirely(self) -> None:
        """spans 中不存在的股票（从未入池）应整体剔除。"""
        data = _make_panel(
            [
                ("000008.SZ", ["2024-01-02", "2024-01-03"]),
                ("600000.SH", ["2024-01-02", "2024-01-03"]),
            ]
        )
        spans = {
            "600000.SH": [(pd.Timestamp("2010-01-01"), pd.Timestamp("2040-12-31"))],
        }
        result = apply_spans_mask(data, spans)

        assert "000008.SZ" not in result.index.get_level_values("instrument")
        assert len(result) == 2

    def test_multi_spans_gap_removed(self) -> None:
        """多段区间：调出-再调入之间的空窗期行被剔除，两段区间内保留。"""
        data = _make_panel(
            [
                (
                    "002074.SZ",
                    ["2018-12-14", "2019-06-03", "2022-06-13", "2022-06-14"],
                ),
            ]
        )
        spans = {
            "002074.SZ": [
                (pd.Timestamp("2016-12-12"), pd.Timestamp("2018-12-14")),
                (pd.Timestamp("2022-06-13"), pd.Timestamp("2040-12-31")),
            ],
        }
        result = apply_spans_mask(data, spans)

        kept = result.index.get_level_values("datetime")
        assert pd.Timestamp("2019-06-03") not in kept, "空窗期应剔除"
        assert len(result) == 3

    def test_empty_data_passthrough(self) -> None:
        """空数据直接原样返回，不报错。"""
        data = _make_panel([("600000.SH", ["2024-01-02"])]).iloc[:0]
        result = apply_spans_mask(data, {"600000.SH": []})

        assert result.empty
