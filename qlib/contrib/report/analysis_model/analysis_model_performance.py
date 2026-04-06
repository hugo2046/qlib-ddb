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
    get_default_init_opts,
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


def compute_group_return(
    pred_label: pd.DataFrame, N: int = 5, reverse: bool = False
) -> tuple:
    """
    计算分组收益

    :param pred_label: 包含 score 和 label 的 DataFrame
    :param N: 分组数，默认为 5
    :param reverse: 是否反转分数
    :return: (group_cum_ret, dist_data, group_avg_ret, group_ret)
        - group_cum_ret: 累计收益 DataFrame
        - dist_data: Long-Short / Long-Average 分布数据 DataFrame
        - group_avg_ret: 各组平均收益 Series
        - group_ret: 日度收益 DataFrame（含 Long-Short、Long-Average）
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

    # 3. 计算多空收益 (Long-Short)
    if not group_ret.empty:
        group_ret["Long-Short"] = group_ret.iloc[:, -1] - group_ret.iloc[:, 0]

    # 4. 计算多均收益 (Long-Average)
    daily_avg = df.groupby("datetime")["label"].mean()
    if not group_ret.empty:
        group_ret["Long-Average"] = group_ret.iloc[:, -1] - daily_avg

    # 5. 累计收益 + 分布数据
    group_cum_ret = group_ret.cumsum()
    dist_data = group_ret[["Long-Short", "Long-Average"]].copy()

    # 6. 画分组柱状图
    group_avg_ret = group_ret.filter(regex="^Group").mean()

    return group_cum_ret, dist_data, group_avg_ret, group_ret


def _group_return(
    pred_label: pd.DataFrame, N: int = 5, reverse: bool = False, config=None, **kwargs
) -> tuple:
    """
    绘制分组累计收益图 + 分布图 + 条形图 + 日度收益日历热力图

    :param pred_label: 包含 score 和 label 的 DataFrame
    :param N: 分组数，默认为 5
    :param reverse: 是否反转分数
    :param config: 可选的配置
    :return: (时序图, 分布图, 条形图, 日历热力图)
    """
    from copy import deepcopy
    from ..display_config import (
        GROUP_RETURN_CONFIG,
        GROUP_RETURN_SUBPLOTS_CONFIG,
        IC_CALENDAR_LAYOUT,
    )
    from ..graph import plot_timeseries, plot_bar, plot_calendar

    # 1. 计算
    group_cum_ret, dist_data, group_avg_ret, group_ret = compute_group_return(
        pred_label, N, reverse
    )

    # 2. 时序图
    config = config or GROUP_RETURN_CONFIG
    graph_ts_fig = plot_timeseries(
        group_cum_ret,
        config=config,
        title="Cumulative Return",
        layout={
            "width": "100%",
            "height": 500,
            "xaxis": {"title": "Date"},
            "yaxis": {"title": "Cumulative Return (Simple Interest)"},
        },
    )

    # 3. 分布图
    _bin_size = float(((dist_data.max() - dist_data.min()) / 20).min())
    subplot_config = deepcopy(GROUP_RETURN_SUBPLOTS_CONFIG)
    subplot_config.kind_map["kwargs"]["bin_size"] = _bin_size

    sub_graph_data = [
        ("Long-Short", dict(row=1, col=1, name="Long-Short", title="Long-Short")),
        ("Long-Average", dict(row=1, col=2, name="Long-Average", title="Long-Average")),
    ]

    graph_hist = SubplotsGraph(
        df=dist_data,
        config=subplot_config,
        sub_graph_data=sub_graph_data,
    )

    # 4. 条形图
    group_avg_ret_df = group_avg_ret.to_frame("Average Return")
    graph_bar_fig = plot_bar(
        group_avg_ret_df,
        config=GROUP_RETURN_CONFIG,
        title="Average Return by Group",
        layout={"width": "100%", "height": 360},
        graph_kwargs={
            "axis_formatter": JsCode(get_axis_percent_formatter(4)),
            "is_show_legend": False,
        },
    )

    # 5. Daily Returns Calendar Heatmap（Long-Short 日度收益日历图）
    _ls_ret = group_ret["Long-Short"].dropna()
    if isinstance(_ls_ret.index, pd.MultiIndex):
        _ls_ret.index = _ls_ret.index.get_level_values("datetime")

    graph_calendar_fig = plot_calendar(
        _ls_ret.to_frame("Long-Short"),
        title="Daily Returns Calendar (Long-Short)",
        layout=IC_CALENDAR_LAYOUT,
        graph_kwargs={
            "visual_map_min": _ls_ret.min(),
            "visual_map_max": _ls_ret.max(),
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
                    "color": ["#10b981", "#6ee7b7", "#f3f4f6", "#fca5a5", "#ef4444"]
                },
            },
        },
    )

    return graph_ts_fig, graph_hist.figure, graph_bar_fig, graph_calendar_fig


def compute_ic(
    pred_label: pd.DataFrame,
    methods: Sequence[str] = ("IC", "Rank IC"),
) -> pd.DataFrame:
    """
    计算原始日度 IC 和 Rank IC 时序

    :param pred_label: 包含 score 和 label 的 DataFrame，MultiIndex (datetime, instrument)
    :param methods: 计算方法列表，默认 ("IC", "Rank IC")
    :return: DataFrame，index 为日期，columns 为各 IC 方法
    """
    _methods_mapping = {"IC": "pearson", "Rank IC": "spearman"}

    def _corr_series(x, method):
        return x["label"].corr(x["score"], method=method)

    # 计算 IC 和 Rank IC
    ic_df = pd.concat(
        [
            pred_label.groupby(level="datetime")
            .apply(partial(_corr_series, method=_methods_mapping[m]))
            .rename(m)
            for m in methods
        ],
        axis=1,
    )

    return ic_df


def _pred_ic(
    pred_label: pd.DataFrame,
    methods: Sequence[str] = ("IC", "Rank IC"),
    config=None,
    **kwargs,
) -> List[object]:
    """
    绘制 IC 分析图 (IC/Rank IC 时序, Daily IC 热力图, IC 分布 + Q-Q 图)

    :param pred_label: 包含 score 和 label 的 DataFrame
    :param methods: 计算方法列表
    :param config: 可选的配置
    :param kwargs: 额外参数：
        - accumulative (bool): 时序图是否使用累积 IC，默认 False。
          Calendar / Distribution / Q-Q 图固定使用原始日度 IC，不受此参数影响。
        - show_nature_day (bool): 是否按自然日填充

    :return: 图表对象列表，顺序为 [时序图, 日历热力图, 分布+QQ组合图]
    """
    from ..graph import plot_timeseries, plot_calendar

    # 1. 计算原始日度 IC（只计算一次）
    show_nature_day = kwargs.get("show_nature_day", False)
    accumulative = kwargs.get("accumulative", False)
    ic_df_raw = compute_ic(pred_label, methods)

    # 2. 时序图数据：根据参数做累积 / 自然日填充后处理
    ic_df = ic_df_raw.copy()
    if accumulative:
        ic_df = ic_df.cumsum()
    if show_nature_day:
        date_index = pd.date_range(ic_df.index.min(), ic_df.index.max())
        ic_df = ic_df.reindex(date_index)

    # 3. IC 时序图（使用后处理后的 ic_df，支持累积模式）
    ic_ts_config = config or MODEL_PERFORMANCE_CONFIG
    graph_ts_fig = plot_timeseries(
        ic_df,
        config=ic_ts_config,
        title="Information Coefficient (IC)",
        layout={"width": "100%"},
    )

    # 4. Daily IC Calendar Heatmap（固定使用原始日度 IC）
    _ic = ic_df_raw.iloc[:, 0]  # 取第一列 (通常是 IC)
    if isinstance(_ic.index, pd.MultiIndex):
        _ic.index = _ic.index.get_level_values("datetime")

    graph_calendar_fig = plot_calendar(
        _ic.to_frame("IC"),
        title="Daily IC Calendar",
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
                    "color": ["#10b981", "#6ee7b7", "#f3f4f6", "#fca5a5", "#ef4444"]
                },
            },
        },
    )

    # 5. IC 分布与 Q-Q 图（固定使用原始日度 IC）
    _ic_df = _ic.to_frame("IC")
    _bin_size = float(((_ic_df.max() - _ic_df.min()) / 20).min())

    graph_dist = DistplotGraph(
        df=_ic_df.dropna(),
        config=IC_DIST_CONFIG,
        layout=IC_DIST_LAYOUT,
        graph_kwargs={"bin_size": _bin_size},
    )

    # Q-Q Plot
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

    # 组合 Grid（使用白色主题）
    grid_ic_qq = (
        Grid(init_opts=get_default_init_opts(width="100%", height=500))
        .add(
            graph_dist.figure,
            grid_opts=opts.GridOpts(pos_left="5%", pos_right="55%", pos_top="15%"),
        )
        .add(
            graph_qq.figure,
            grid_opts=opts.GridOpts(pos_left="55%", pos_right="5%", pos_top="15%"),
        )
    )

    return [graph_ts_fig, graph_calendar_fig, grid_ic_qq]


def compute_autocorr(pred_label: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """
    计算预测值自相关性（Rank Correlation 时序）

    Args:
        pred_label: 包含 score 和 label 的 DataFrame，MultiIndex (datetime, instrument)
        lag: 滞后期数，默认为 1

    Returns:
        DataFrame，index 为日期，列为 value（自相关系数）
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
    return ac.to_frame("value")


def _pred_autocorr(
    pred_label: pd.DataFrame, lag: int = 1, config=None, **kwargs
) -> List[object]:
    """
    绘制预测值自相关性（Rank Correlation 时序图）

    Args:
        pred_label: 包含 score 和 label 的 DataFrame
        lag: 滞后期数，默认为 1
        config: 可选的 GraphDisplayConfig 配置

    Returns:
        pyecharts 图表对象列表
    """
    from ..display_config import AUTOCORR_CONFIG
    from ..graph import plot_timeseries

    # 1. 计算
    df = compute_autocorr(pred_label, lag)

    # 2. 可视化
    config = config or AUTOCORR_CONFIG
    fig = plot_timeseries(
        df,
        config=config,
        title=f"Auto Correlation (Lag={lag})",
        layout={
            "width": "100%",
            "height": 400,
            "xaxis": {"title": "Date"},
            "yaxis": {"title": "Rank Correlation"},
        },
    )

    return [fig]


def compute_turnover(
    pred_label: pd.DataFrame, N: int = 5, lag: int = 1
) -> pd.DataFrame:
    """
    计算 Top/Bottom 组合换手率 (基于分位数分组)

    Turnover = 1 - (此期组合与上期组合重合数 / 组合总数)

    Args:
        pred_label: 包含 score 和 label 的 DataFrame
        N: 分组数，默认为 5 (即 Top/Bottom 20%)
        lag: 滞后期数，默认为 1

    Returns:
        DataFrame，index 为日期，columns 为 ["Top", "Bottom"]
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

    return pd.DataFrame(turnover_data)


def _pred_turnover(
    pred_label: pd.DataFrame,
    N: int = 5,
    lag: int = 1,
    config=None,
    **kwargs,
) -> List[object]:
    """
    绘制 Top/Bottom 组合换手率

    Args:
        pred_label: 包含 score 和 label 的 DataFrame
        N: 分组数，默认为 5 (即 Top/Bottom 20%)
        lag: 滞后期数，默认为 1
        config: 可选的 GraphDisplayConfig 配置

    Returns:
        pyecharts 图表对象列表
    """
    from ..display_config import TURNOVER_CONFIG
    from ..graph import plot_timeseries

    # 1. 计算
    df = compute_turnover(pred_label, N, lag)

    # 2. 可视化
    config = config or TURNOVER_CONFIG
    fig = plot_timeseries(
        df,
        config=config,
        title=f"Top-Bottom Turnover (Lag={lag}, N={N})",
        layout={
            "width": "100%",
            "height": 400,
            "xaxis": {"title": "Date"},
            "yaxis": {"title": "Turnover Rate"},
        },
    )

    return [fig]


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
    cumulative_ic: bool = False,
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
    :param cumulative_ic: 是否绘制累积 IC 曲线（cumsum），仅当 graph_names 包含 'pred_ic' 时生效
    :param graph_names: 图表名称列表；默认 ['group_return', 'pred_ic', 'pred_autocorr']
    :param show_notebook: 是否在 Notebook 中显示
    :param **kwargs: 额外参数（兼容原始接口）
    :return: 如果 show_notebook 为 True，在 notebook 中显示；否则返回图表列表
    """

    # 使用与原始版本相同的实现方式：遍历 graph_names 动态调用函数
    figure_list = []
    # 将 cumulative_ic 注入 kwargs，传递给 _pred_ic 中的 compute_ic（参数名映射为 accumulative）
    kwargs.setdefault("accumulative", cumulative_ic)
    if cumulative_ic and "pred_ic" not in graph_names:
        import warnings

        warnings.warn(
            "cumulative_ic=True 仅对 'pred_ic' 有效，但 graph_names 中未包含 'pred_ic'，该参数不会生效。",
            UserWarning,
            stacklevel=2,
        )
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
