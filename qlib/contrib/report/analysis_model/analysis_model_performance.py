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
    get_number_formatter,
    get_axis_percent_formatter,
)
from ..display_config import (
    GROUP_RETURN_SUBPLOTS_CONFIG,
    MODEL_PERFORMANCE_CONFIG,
    IC_HEATMAP_LAYOUT,
    IC_DIST_LAYOUT,
    IC_QQ_LAYOUT,
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

    # 1. 分箱
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
    # 参考 Plotly 版本逻辑: Group1 - GroupN
    if not group_ret.empty:
        group_ret["Long-Short"] = group_ret.iloc[:, 0] - group_ret.iloc[:, -1]

    # 4. 计算多均收益 (Long-Average) - 基于日收益率计算
    daily_avg = df.groupby("datetime")["label"].mean()
    if not group_ret.empty:
        # Long-Average = Group1 - Daily Average
        group_ret["Long-Average"] = group_ret.iloc[:, 0] - daily_avg

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

    # 3. Monthly IC 热力图
    # 取第一列 (通常是 IC) 计算月度均值
    _ic = ic_df.iloc[:, 0]

    # 计算月度 IC (按 YYYYMM 分组)
    _index = (
        _ic.index.get_level_values(0).astype("str").str.replace("-", "").str.slice(0, 6)
    )
    _monthly_ic = _ic.groupby(_index).mean()

    # 重构 MultiIndex (Year, Month)
    _monthly_ic.index = pd.MultiIndex.from_arrays(
        [_monthly_ic.index.str.slice(0, 4), _monthly_ic.index.str.slice(4, 6)],
        names=["year", "month"],
    )

    # 填充缺失月份
    start_year = _index.min()[:4]
    end_year = _index.max()[:4]
    # 使用 'M' 作为频率 (MonthEnd)
    _month_list = pd.date_range(f"{start_year}0101", f"{end_year}1231", freq="M")
    _years = [d.strftime("%Y") for d in _month_list]
    _months = [d.strftime("%m") for d in _month_list]
    fill_index = pd.MultiIndex.from_arrays([_years, _months], names=["year", "month"])

    _monthly_ic = _monthly_ic.reindex(fill_index)

    # 热力图: Unstack 后 Index=Year, Columns=Month
    graph_heatmap = HeatmapGraph(
        _monthly_ic.unstack(),
        layout=IC_HEATMAP_LAYOUT,
        graph_kwargs={
            "visual_map_min": _monthly_ic.min(),
            "visual_map_max": _monthly_ic.max(),
        },
    )

    # 4. IC 分布与 Q-Q 图
    # 4.1 分布图
    _ic_df = _ic.to_frame("IC")
    _bin_size = float(((_ic_df.max() - _ic_df.min()) / 20).min())

    graph_dist = DistplotGraph(
        df=_ic_df.dropna(),
        layout=IC_DIST_LAYOUT,
        graph_kwargs={"bin_size": _bin_size},
    )

    # 4.2 Q-Q Plot
    qq_figure = _plot_qq(_ic, show_notebook=False)

    # 4.3 组合在一个 Grid 中 (类似于 SubplotsGraph 效果)
    # 手动调整子图的 Title 位置和 Legend
    # Dist Plot: Left 5% ~ 45% (Width 40%), Center ~ 25%
    graph_dist.chart.set_global_opts(
        legend_opts=opts.LegendOpts(is_show=False),
        title_opts=opts.TitleOpts(title="IC Distribution", pos_left="25%"),
    )

    # QQ Plot: Left 55% ~ 95% (Width 40%), Center ~ 75%
    qq_figure.set_global_opts(
        legend_opts=opts.LegendOpts(is_show=False),
        title_opts=opts.TitleOpts(title="IC Normal Dist. Q-Q", pos_left="75%"),
    )

    grid_hist_qq = (
        Grid(init_opts=opts.InitOpts(width="100%", height="500px"))
        .add(
            graph_dist.figure,
            grid_opts=opts.GridOpts(pos_left="5%", pos_right="55%", pos_top="20%"),
        )
        .add(
            qq_figure,
            grid_opts=opts.GridOpts(pos_left="55%", pos_right="5%", pos_top="20%"),
        )
    )

    figs = [graph_ts.figure, graph_heatmap.figure, grid_hist_qq]

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
