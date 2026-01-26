"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2026-01-18 15:30:02
LastEditors: shen.lan123@gmail.com
LastEditTime: 2026-01-25 21:24:24
Description: 使用pyecharts重构模型表现分析图表
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Union
from pyecharts.commons.utils import JsCode

# 引入 Pyecharts 组件
from pyecharts.charts import Line, Bar, Grid
from pyecharts import options as opts

# 引入我们封装好的 graph 组件 (请确保 graph.py 路径正确)
from qlib.contrib.report.graph import (
    BarGraph,
    BaseGraph,
    DistplotGraph,
    ScatterGraph,
    SubplotsGraph,
    get_number_formatter,
    get_percent_formatter,
    get_axis_percent_formatter,
)
from ..display_config import GROUP_RETURN_SUBPLOTS_CONFIG

# ==============================================================================
# 私有辅助函数 (子组件重构)
# 这些函数负责具体的绘图逻辑，被主函数调用
# ==============================================================================


def _plot_qq(score: pd.Series, show_notebook: bool = True) -> object:
    """
    绘制 QQ 图 (Quantile-Quantile Plot)
    """
    # 1. 计算 QQ 数据
    (osm, osr), (slope, intercept, r) = stats.probplot(score, dist="norm")

    # 2. 构造数据
    df_qq = pd.DataFrame({"Sample Quantiles": osr}, index=osm)
    # 添加参考线
    df_qq["Reference Line"] = df_qq.index * slope + intercept

    # 3. 绘图
    # 技巧：通过 layout 强制指定 X 轴为数值轴
    graph = ScatterGraph(
        df=df_qq,
        layout={
            "title": "Normal Q-Q Plot",
            "width": "100%",
            "height": 500,
            "xaxis": {"type": "value", "title": "Theoretical Quantiles"},
            "yaxis": {"title": "Sample Quantiles"},
        },
        graph_kwargs={"mode": "lines", "is_symbol_show": False},  # 用线模式模拟密集点
    )

    if show_notebook:
        ScatterGraph.show_graph_in_notebook([graph.figure])
    return graph.figure


def _group_return(
    pred_label: pd.DataFrame, N: int = 5, reverse: bool = False, **kwargs
) -> tuple:
    """
    绘制分组累计收益图 + 分布图

    返回: (group_scatter_figure, group_hist_figure)
    """
    df = pred_label.copy()
    if reverse:
        df["score"] *= -1

    # 1. 分箱
    def get_group(x):
        try:
            return pd.qcut(x, N, labels=False, duplicates="drop")
        except ValueError:
            return np.nan

    df["group"] = df.groupby("datetime")["score"].transform(get_group)

    # 2. 计算各组收益 (单利累加)
    group_ret = df.groupby(["datetime", "group"])["label"].mean().unstack()
    group_cum_ret = group_ret.cumsum()
    group_cum_ret.columns = [f"Group{i+1}" for i in range(len(group_cum_ret.columns))]

    # 3. 计算多空收益 (Long-Short)
    if not group_cum_ret.empty:
        group_cum_ret["Long-Short"] = (
            group_cum_ret.iloc[:, -1] - group_cum_ret.iloc[:, 0]
        )

    # 4. 计算多均收益 (Long-Average)
    daily_avg = df.groupby("datetime")["label"].mean()
    if not group_cum_ret.empty and len(group_cum_ret) > 0:
        group_cum_ret["Long-Average"] = group_cum_ret.iloc[:, 0] - daily_avg

    # 5. 绘制时序图
    from ..display_config import GROUP_RETURN_CONFIG

    graph_ts = ScatterGraph(
        df=group_cum_ret,
        config=GROUP_RETURN_CONFIG,
        layout={
            "title": "Cumulative Return",
            "width": "100%",
            "height": 500,
            "xaxis": {"title": "Date"},
            "yaxis": {"title": "Cumulative Return (Simple Interest)"},
        },
        graph_kwargs={
            "mode": "lines",
        },
    )

    # 6. 准备分布图数据 (long-short 和 long-average)
    dist_data = group_cum_ret[["Long-Short", "Long-Average"]].copy()
    _bin_size = float(((dist_data.max() - dist_data.min()) / 20).min())

    # 7. 绘制分布图 (使用 SubplotsGraph)
    # Update kwargs for DistplotGraph dynamically
    from copy import deepcopy

    config = deepcopy(GROUP_RETURN_SUBPLOTS_CONFIG)
    config.kind_map["kwargs"]["bin_size"] = _bin_size

    # 构建 sub_graph_data 以指定每个子图的标题
    sub_graph_data = [
        ("Long-Short", dict(row=1, col=1, name="Long-Short", title="Long-Short")),
        ("Long-Average", dict(row=1, col=2, name="Long-Average", title="Long-Average")),
    ]

    graph_hist = SubplotsGraph(
        df=dist_data,
        config=config,
        sub_graph_data=sub_graph_data,
    )

    return graph_ts.figure, graph_hist.figure


def _pred_ic(
    pred_label: pd.DataFrame, rank: bool = False, show_notebook: bool = True, **kwargs
) -> List[object]:
    """
    绘制 IC 分析图 (时序图 + 分布图)
    """

    # 1. 计算每日 IC
    def calc_ic(x):
        if rank:
            return x["score"].rank().corr(x["label"].rank())
        else:
            return x["score"].corr(x["label"])

    daily_ic = pred_label.groupby("datetime").apply(calc_ic)

    # 2. 准备数据
    df_ic_ts = pd.DataFrame(
        {"Daily IC": daily_ic, "IC MA20": daily_ic.rolling(20).mean()}
    )

    # 3. 绘图
    # 图A: IC 时序
    from ..display_config import MODEL_PERFORMANCE_CONFIG

    graph_ts = ScatterGraph(
        df=df_ic_ts,
        config=MODEL_PERFORMANCE_CONFIG,
        layout={
            "title": "Information Coefficient (IC)",
            "width": "100%",
        },
        graph_kwargs={
            "mode": "lines",
        },
    )

    # 图B: IC 分布
    graph_dist = DistplotGraph(
        df=daily_ic.to_frame("IC"),
        layout={"title": "IC Distribution", "width": "100%", "height": 400},
    )

    figs = [graph_ts.figure, graph_dist.figure]

    if show_notebook:
        BaseGraph.show_graph_in_notebook(figs)
    return figs


def _pred_autocorr(
    pred_label: pd.DataFrame, lag: int = 1, show_notebook: bool = True
) -> object:
    """
    绘制预测值自相关性
    """
    score = pred_label["score"]

    # 1. 计算 Lag 1-5 的自相关系数
    ac_data = {}
    lags = range(1, 6)

    # 将数据转为 Panel 格式 (Index: date, Columns: instrument) 以加速计算
    try:
        score_unstack = score.unstack(level="instrument")
    except Exception:
        score_unstack = score.reset_index().pivot(
            index="datetime", columns="instrument", values="score"
        )

    for l in lags:
        # 计算每只股票的自相关系数，取平均
        ac = score_unstack.apply(lambda x: x.autocorr(lag=l)).mean()
        ac_data[f"Lag-{l}"] = ac

    df_ac = pd.DataFrame(
        list(ac_data.values()), index=list(ac_data.keys()), columns=["Autocorr"]
    )

    # 2. 绘图 (柱状图)
    graph = BarGraph(
        df=df_ac,
        layout={
            "title": "Prediction Autocorrelation",
            "width": "100%",
            "height": 400,
            "yaxis": {"title": "Autocorrelation Coefficient"},
        },
    )

    if show_notebook:
        BarGraph.show_graph_in_notebook([graph.figure])
    return graph.figure


# ==============================================================================
# 主入口函数 (API)
# 用户只需要调用这个函数
# ==============================================================================


def model_performance_graph(
    pred_label: pd.DataFrame,
    lag: int = 1,
    N: int = 5,
    reverse: bool = False,
    rank: bool = False,
    graph_names: list = ["group_return", "pred_ic", "pred_autocorr"],
    show_notebook: bool = True,
    **kwargs,
) -> List[object]:
    """
    生成完整的模型性能分析报告

    :param pred_label: 包含 score 和 label 的 DataFrame
    :param lag: 滞后天数
    :param N: 分组数
    :param reverse: 是否反转分数
    :param rank: 是否使用 Rank IC
    :param graph_names: 图表名称列表；默认 ['group_return', 'pred_ic', 'pred_autocorr']
    :param show_notebook: 是否在 Notebook 中显示
    :param **kwargs: 额外参数（兼容原始接口）
    :return: 如果 show_notebook 为 True，在 notebook 中显示；否则返回图表列表
    """

    # 使用与原始版本相同的实现方式：遍历 graph_names 动态调用函数
    figure_list = []
    for graph_name in graph_names:
        # 动态调用函数（如 _group_return, _pred_ic, _pred_autocorr）
        # 注意：eval() 与原始版本保持一致，用于动态函数调用
        fun_res = eval(f"_{graph_name}")(
            pred_label=pred_label, lag=lag, N=N, reverse=reverse, rank=rank, **kwargs
        )
        figure_list += fun_res

    if show_notebook:
        BaseGraph.show_graph_in_notebook(figure_list)
    else:
        return figure_list
