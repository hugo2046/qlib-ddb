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
import statsmodels.api as sm
import matplotlib.pyplot as plt
from typing import List, Union, Sequence
from functools import partial
from pyecharts.commons.utils import JsCode

# 引入 Pyecharts 组件
from pyecharts.charts import Line, Bar, Grid
from pyecharts import options as opts

# 引入我们封装好的 graph 组件 (请确保 graph.py 路径正确)
from qlib.contrib.report.graph import (
    BarGraph,
    HeatmapGraph,
    QQPlotGraph,
    BaseGraph,
    DistplotGraph,
    ScatterGraph,
    SubplotsGraph,
    CalendarGraph,
    get_number_formatter,
    get_calendar_formatter,
    get_axis_percent_formatter,
    get_percent_formatter,
)
from ..display_config import (
    GROUP_RETURN_SUBPLOTS_CONFIG,
    GROUP_RETURN_CONFIG,
    MODEL_PERFORMANCE_CONFIG,
    IC_HEATMAP_LAYOUT,
    IC_DIST_LAYOUT,
    IC_QQ_LAYOUT,
    IC_DIST_CONFIG,
    IC_SUBPLOTS_CONFIG,
    IC_QQ_CONFIG,
    IC_CALENDAR_LAYOUT,
)

# ==============================================================================
# 私有辅助函数 (子组件重构)
# 这些函数负责具体的绘图逻辑，被主函数调用
# ==============================================================================


def _plot_qq(score: pd.Series, show_notebook: bool = True) -> object:
    """
    绘制 QQ 图 (Quantile-Quantile Plot)
    使用 statsmodels 计算数据以保持与旧版一致
    """
    # 1. 计算 QQ 数据 (复用 bak.py 逻辑)
    # fit=True 会拟合数据的 loc 和 scale
    _plt_fig = sm.qqplot(score.dropna(), dist=stats.norm, fit=True, line="45")
    plt.close(_plt_fig)

    qqplot_data = _plt_fig.gca().lines
    # trace 0 is sample data (markers)
    x_data = qqplot_data[0].get_xdata()
    y_data = qqplot_data[0].get_ydata()

    # 2. 构造数据
    # QQPlotGraph 使用 Index 作为 X (Theoretical), Column 0 作为 Y (Sample)
    df_qq = pd.DataFrame({"Sample Quantiles": y_data}, index=x_data)

    # 3. 绘图
    # 使用专门处理数值轴的 QQPlotGraph
    graph = QQPlotGraph(
        df=df_qq,
        layout=IC_QQ_LAYOUT,
    )

    if show_notebook:
        BaseGraph.show_graph_in_notebook([graph.figure])
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

    # 1. 分箱-升序排列
    def get_group(x):
        try:
            return pd.qcut(x, N, labels=False, duplicates="drop")
        except ValueError:
            return np.nan

    df["group"] = df.groupby("datetime")["score"].transform(get_group)

    # 2. 计算各组收益 (单利) - 计算日收益率
    group_ret = df.groupby(["datetime", "group"])["label"].mean().unstack()
    group_ret.columns = [f"Group{i+1}" for i in range(len(group_ret.columns))]

    # 3. 计算多空收益 (Long-Short) - 基于日收益率计算
    # 分组默认升序排列top组-bottom组
    if not group_ret.empty:
        group_ret["Long-Short"] = group_ret.iloc[:, -1] - group_ret.iloc[:, 0]

    # 4. 计算多均收益 (Long-Average) - 基于日收益率计算
    daily_avg = df.groupby("datetime")["label"].mean()
    if not group_ret.empty:
        # Long-Average = Top Group - Daily Average
        group_ret["Long-Average"] = group_ret.iloc[:, -1] - daily_avg

    # 5. 计算累计收益 (用于绘制时序图)
    group_cum_ret = group_ret.cumsum()

    # 绘制时序图使用 group_cum_ret
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

    # 6. 准备分布图数据 (long-short 和 long-average) - 使用日收益率
    dist_data = group_ret[["Long-Short", "Long-Average"]].copy()
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
    pred_label: pd.DataFrame,
    methods: Sequence[str] = ("IC", "Rank IC"),
    **kwargs,
) -> List[object]:
    """
    绘制 IC 分析图 (IC/Rank IC 时序, Monthly IC 热力图, IC 分布 + Q-Q 图)
    """
    _methods_mapping = {"IC": "pearson", "Rank IC": "spearman"}

    def _corr_series(x, method):
        return x["label"].corr(x["score"], method=method)

    # 1. 计算 IC 和 Rank IC
    # 使用 level="datetime" 以兼容 MultiIndex
    ic_df = pd.concat(
        [
            pred_label.groupby(level="datetime")
            .apply(partial(_corr_series, method=_methods_mapping[m]))
            .rename(m)
            for m in methods
        ],
        axis=1,
    )

    # 2. IC 时序图
    if kwargs.get("show_nature_day", False):
        date_index = pd.date_range(ic_df.index.min(), ic_df.index.max())
        ic_df_reindexed = ic_df.reindex(date_index)
    else:
        ic_df_reindexed = ic_df

    # 使用 Line 模式绘制 IC 时序 (ScatterGraph)
    graph_ts = ScatterGraph(
        df=ic_df_reindexed,
        config=MODEL_PERFORMANCE_CONFIG,
        layout={
            "title": "Information Coefficient (IC)",
            "width": "100%",
        },
        graph_kwargs={
            "mode": "lines",
        },
    )

    # 3. Daily IC Calendar Heatmap
    # 取第一列 (通常是 IC) 的日度数据
    _ic = ic_df.iloc[:, 0]

    # 确保 index 是 DatetimeIndex（处理可能的 MultiIndex）
    if isinstance(_ic.index, pd.MultiIndex):
        # 提取 datetime level
        _ic.index = _ic.index.get_level_values("datetime")

    # 直接使用日度 IC 数据创建 Calendar Heatmap
    graph_calendar = CalendarGraph(
        df=_ic.to_frame("IC"),
        layout=IC_CALENDAR_LAYOUT,
        graph_kwargs={
            "visual_map_min": _ic.min(),
            "visual_map_max": _ic.max(),
            "tooltip": {
                "position": "top",
                "formatter": JsCode(get_calendar_formatter(4)),
            },
            "visualMap": {
                "calculable": True,
                "orient": "horizontal",
                "left": "75%",
                "top": "top",
                "inRange": {
                    # A股审美：负值绿色（跌），正值红色（涨）
                    # 使用现代前端配色：柔和渐变 + 高对比度
                    "color": ["#10b981", "#6ee7b7", "#f3f4f6", "#fca5a5", "#ef4444"]
                },
            },
        },
    )

    # 4. IC 分布与 Q-Q 图 (使用 Grid 组合两个独立图表)
    # 4.1 IC Distribution
    _ic_df = _ic.to_frame("IC")
    _bin_size = float(((_ic_df.max() - _ic_df.min()) / 20).min())

    graph_dist = DistplotGraph(
        df=_ic_df.dropna(),
        config=IC_DIST_CONFIG,
        layout=IC_DIST_LAYOUT,
        graph_kwargs={"bin_size": _bin_size},
    )

    # 4.2 Q-Q Plot (不使用 _plot_qq，直接在这里创建)
    _plt_fig = sm.qqplot(_ic.dropna(), dist=stats.norm, fit=True, line="45")
    plt.close(_plt_fig)
    qqplot_data = _plt_fig.gca().lines
    x_data = qqplot_data[0].get_xdata()
    y_data = qqplot_data[0].get_ydata()
    df_qq = pd.DataFrame({"Sample Quantiles": y_data}, index=x_data)

    graph_qq = QQPlotGraph(
        df=df_qq,
        config=IC_QQ_CONFIG,
        layout=IC_QQ_LAYOUT,
    )

    # 4.3 使用 Grid 组合
    grid_ic_qq = (
        Grid(init_opts=opts.InitOpts(width="100%", height="500px"))
        .add(
            graph_dist.figure,
            grid_opts=opts.GridOpts(pos_left="5%", pos_right="55%", pos_top="15%"),
        )
        .add(
            graph_qq.figure,
            grid_opts=opts.GridOpts(pos_left="55%", pos_right="5%", pos_top="15%"),
        )
    )

    figs = [graph_ts.figure, graph_calendar.figure, grid_ic_qq]

    return figs


def _pred_autocorr(pred_label: pd.DataFrame, lag: int = 1, **kwargs) -> List[object]:
    """
    绘制预测值自相关性（Rank Correlation 时序图）

    计算每日预测值与滞后预测值的 Spearman 秩相关系数，
    并绘制时序散点图展示自相关随时间的变化趋势。

    Args:
        pred_label: 包含 score 和 label 的 DataFrame，
                   MultiIndex (datetime, instrument)
        lag: 滞后期数，默认为 1
        show_notebook: 是否在 notebook 中显示

    Returns:
        pyecharts 图表对象
    """
    pred = pred_label.copy()

    # 1. 计算滞后值（按股票分组）
    pred["score_last"] = pred.groupby(level="instrument", group_keys=False)[
        "score"
    ].shift(lag)

    # 2. 按日期分组，计算每日的 Rank Correlation
    ac = pred.groupby(level="datetime", group_keys=False).apply(
        lambda x: x["score"].corr(x["score_last"], method="spearman")
    )

    # 3. 转换为 DataFrame
    _df = ac.to_frame("value")

    # 4. 使用 ScatterGraph 绘制时序图
    graph = ScatterGraph(
        df=_df,
        layout={
            "title": f"Auto Correlation (Lag={lag})",
            "width": "100%",
            "height": 400,
            "xaxis": {"title": "Date"},
            "yaxis": {"title": "Rank Correlation"},
        },
        graph_kwargs={
            "mode": "lines",
            "is_show_legend": False,
            "tooltip_formatter": JsCode(get_number_formatter(4)),
        },
    )

    return [graph.figure]


def _pred_turnover(
    pred_label: pd.DataFrame,
    N: int = 5,
    lag: int = 1,
    **kwargs,  # 接收可能的额外参数
) -> List[object]:
    """
    绘制 Top/Bottom 组合换手率 (基于分位数分组)

    Turnover = 1 - (此期组合与上期组合重合数 / 组合总数)
    衡量 Alpha 信号的稳定性及潜在交易成本。

    Args:
        pred_label: 包含 score 和 label 的 DataFrame
        N: 分组数，默认为 5 (即 Top/Bottom 20%)
        lag: 滞后期数，默认为 1
        show_notebook: 是否在 notebook 中显示

    Returns:
        pyecharts 图表对象
    """
    pred = pred_label.copy()
    score = pred["score"]

    # 1. 分组 (Quantile Binning)
    def get_group(x):
        try:
            return pd.qcut(x, N, labels=False, duplicates="drop")
        except ValueError:
            return np.nan

    # 计算每日的分组 (0 到 N-1)
    groups = score.groupby(level="datetime", group_keys=False).apply(get_group)

    # 2. 转换为宽表加速计算
    try:
        group_df = groups.unstack(level="instrument")
    except Exception:
        group_df = groups.reset_index().pivot(
            index="datetime", columns="instrument", values="score"
        )

    # 3. 计算 Top/Bottom 换手率
    group_df_last = group_df.shift(lag)
    top_group = N - 1
    bottom_group = 0

    turnover_data = {}

    for name, g_id in [("Top", top_group), ("Bottom", bottom_group)]:
        mask_curr = group_df == g_id
        mask_last = group_df_last == g_id
        count_curr = mask_curr.sum(axis=1)
        overlap = (mask_curr & mask_last).sum(axis=1)
        turnover = 1 - (overlap / count_curr)
        turnover_data[name] = turnover.fillna(0)

    df_turnover = pd.DataFrame(turnover_data)

    # 4. 绘图
    graph = ScatterGraph(
        df=df_turnover,
        layout={
            "title": f"Top-Bottom Turnover (Lag={lag}, N={N})",
            "width": "100%",
            "height": 400,
            "xaxis": {"title": "Date"},
            "yaxis": {"title": "Turnover Rate"},
        },
        graph_kwargs={
            "mode": "lines",
            "is_show_legend": True,
            "legend_pos_left": "75%",
            "tooltip_formatter": JsCode(get_percent_formatter(2)),
            "axis_formatter": JsCode(get_axis_percent_formatter(2)),
        },
    )

    return [graph.figure]  # 返回列表以适配 model_performance_graph


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
    graph_names: list = ["group_return", "pred_ic", "pred_autocorr", "pred_turnover"],
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
