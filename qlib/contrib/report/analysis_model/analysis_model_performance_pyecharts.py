# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from functools import partial

import pandas as pd

import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy import stats

from typing import Sequence
from qlib.typehint import Literal

from ..pyecharts_graph import (
    ScatterPyechartsGraph, SubplotsPyechartsGraph, BarPyechartsGraph,
    HeatmapPyechartsGraph, HistogramPyechartsGraph, LinePyechartsGraph, show_graph_in_notebook
)
from ..utils import guess_plotly_rangebreaks


def _group_return_pyecharts(pred_label: pd.DataFrame = None, reverse: bool = False, N: int = 5, **kwargs) -> tuple:
    """
    使用pyecharts绘制分组收益图

    :param pred_label:
    :param reverse:
    :param N:
    :return:
    """
    if reverse:
        pred_label["score"] *= -1

    pred_label = pred_label.sort_values("score", ascending=False)

    # Group1 ~ Group5 only consider the dropna values
    pred_label_drop = pred_label.dropna(subset=["score"])

    # Group
    t_df = pd.DataFrame(
        {
            "Group%d"
            % (i + 1): pred_label_drop.groupby(level="datetime")["label"].apply(
                lambda x: x[len(x) // N * i : len(x) // N * (i + 1)].mean()  # pylint: disable=W0640
            )
            for i in range(N)
        }
    )
    t_df.index = pd.to_datetime(t_df.index)

    # Long-Short
    t_df["long-short"] = t_df["Group1"] - t_df["Group%d" % N]

    # Long-Average
    t_df["long-average"] = t_df["Group1"] - pred_label.groupby(level="datetime")["label"].mean()

    t_df = t_df.dropna(how="all")  # for days which does not contain label

    # Cumulative Return By Group
    t_df_cumsum = t_df.cumsum()
    group_scatter_figure = ScatterPyechartsGraph(
        t_df_cumsum,
        title="Cumulative Return",
        width="1000px",
        height="600px"
    ).chart

    t_df = t_df.loc[:, ["long-short", "long-average"]]
    _bin_size = float(((t_df.max() - t_df.min()) / 20).min())

    # 创建子图 - 使用直方图替代分布图
    group_hist_figure = SubplotsPyechartsGraph(
        t_df,
        kind_map=dict(kind="HistogramPyechartsGraph", kwargs=dict(bin_size=_bin_size)),
        title="Distribution Analysis",
        width="1000px",
        height="500px",
        rows=1,
        cols=2,
        subplot_titles=["long-short", "long-average"]
    )

    return group_scatter_figure, group_hist_figure


def _plot_qq_pyecharts(data: pd.Series = None, dist=stats.norm):
    """
    使用pyecharts绘制Q-Q图

    :param data:
    :param dist:
    :return:
    """
    # 使用statsmodels生成Q-Q图数据
    _plt_fig = sm.qqplot(data.dropna(), dist=dist, fit=True, line="45")
    plt.close(_plt_fig)
    qqplot_data = _plt_fig.gca().lines

    # 准备数据
    theoretical_quantiles = qqplot_data[0].get_xdata()
    sample_quantiles = qqplot_data[0].get_ydata()
    line_x = qqplot_data[1].get_xdata()
    line_y = qqplot_data[1].get_ydata()

    # 确保所有数组长度一致
    min_length = min(len(theoretical_quantiles), len(sample_quantiles), len(line_x), len(line_y))

    # 创建DataFrame用于绘图
    qq_df = pd.DataFrame({
        'Theoretical': theoretical_quantiles[:min_length],
        'Sample': sample_quantiles[:min_length],
        'Line_x': line_x[:min_length],
        'Line_y': line_y[:min_length]
    })

    from ..pyecharts_graph import LinePyechartsGraph, ScatterPyechartsGraph

    # 创建散点图
    scatter_chart = ScatterPyechartsGraph(
        qq_df[['Theoretical', 'Sample']],
        title="Q-Q Plot",
        width="600px",
        height="500px"
    ).chart

    # 添加参考线
    scatter_chart.add_xaxis(qq_df['Line_x'].tolist())
    scatter_chart.add_yaxis(
        series_name="Reference Line",
        y_axis=qq_df['Line_y'].tolist(),
        symbol_size=1,
        color="#636efa"
    )

    del qqplot_data
    return scatter_chart


def _pred_ic_pyecharts(
    pred_label: pd.DataFrame = None, methods: Sequence[Literal["IC", "Rank IC"]] = ("IC", "Rank IC"), **kwargs
) -> tuple:
    """
    使用pyecharts绘制预测IC分析图

    :param pred_label: pd.DataFrame
    must contain one column of realized return with name `label` and one column of predicted score names `score`.
    :param methods: Sequence[Literal["IC", "Rank IC"]]
    IC series to plot.
    IC is sectional pearson correlation between label and score
    Rank IC is the spearman correlation between label and score
    For the Monthly IC, IC histogram, IC Q-Q plot.  Only the first type of IC will be plotted.
    :return:
    """
    _methods_mapping = {"IC": "pearson", "Rank IC": "spearman"}

    def _corr_series(x, method):
        return x["label"].corr(x["score"], method=method)

    ic_df = pd.concat(
        [
            pred_label.groupby(level="datetime").apply(partial(_corr_series, method=_methods_mapping[m])).rename(m)
            for m in methods
        ],
        axis=1,
    )
    _ic = ic_df.iloc(axis=1)[0]

    _index = _ic.index.get_level_values(0).astype("str").str.replace("-", "").str.slice(0, 6)
    _monthly_ic = _ic.groupby(_index).mean()
    _monthly_ic.index = pd.MultiIndex.from_arrays(
        [_monthly_ic.index.str.slice(0, 4), _monthly_ic.index.str.slice(4, 6)],
        names=["year", "month"],
    )

    # fill month
    _month_list = pd.date_range(
        start=pd.Timestamp(f"{_index.min()[:4]}0101"),
        end=pd.Timestamp(f"{_index.max()[:4]}1231"),
        freq="1M",
    )
    _years = []
    _month = []
    for _date in _month_list:
        _date = _date.strftime("%Y%m%d")
        _years.append(_date[:4])
        _month.append(_date[4:6])

    fill_index = pd.MultiIndex.from_arrays([_years, _month], names=["year", "month"])

    _monthly_ic = _monthly_ic.reindex(fill_index)

    ic_bar_figure = ic_figure_pyecharts(ic_df, kwargs.get("show_nature_day", False))

    ic_heatmap_figure = HeatmapPyechartsGraph(
        _monthly_ic.unstack(),
        title="Monthly IC",
        width="800px",
        height="600px"
    ).chart

    dist = stats.norm
    _qqplot_fig = _plot_qq_pyecharts(_ic, dist)

    if isinstance(dist, stats.norm.__class__):
        dist_name = "Normal"
    else:
        dist_name = "Unknown"

    _ic_df = _ic.to_frame("IC")
    _bin_size = ((_ic_df.max() - _ic_df.min()) / 20).min()

    # 创建包含直方图和Q-Q图的页面
    ic_hist_figure = SubplotsPyechartsGraph(
        _ic_df.dropna(),
        kind_map=dict(kind="HistogramPyechartsGraph", kwargs=dict(bin_size=_bin_size)),
        title="IC Distribution Analysis",
        width="1200px",
        height="500px",
        rows=1,
        cols=2,
        subplot_titles=["IC", f"IC {dist_name} Dist. Q-Q"]
    )

    return ic_bar_figure, ic_heatmap_figure, ic_hist_figure, _qqplot_fig


def _pred_autocorr_pyecharts(pred_label: pd.DataFrame, lag=1, **kwargs) -> tuple:
    """使用pyecharts绘制预测自相关图"""
    pred = pred_label.copy()
    pred["score_last"] = pred.groupby(level="instrument")["score"].shift(lag)
    ac = pred.groupby(level="datetime").apply(lambda x: x["score"].rank(pct=True).corr(x["score_last"].rank(pct=True)))
    _df = ac.to_frame("value")
    ac_figure = LinePyechartsGraph(
        _df,
        title="Auto Correlation",
        width="1000px",
        height="500px"
    ).chart
    return (ac_figure,)


def _pred_turnover_pyecharts(pred_label: pd.DataFrame, N=5, lag=1, **kwargs) -> tuple:
    """使用pyecharts绘制换手率图"""
    pred = pred_label.copy()
    pred["score_last"] = pred.groupby(level="instrument")["score"].shift(lag)
    top = pred.groupby(level="datetime").apply(
        lambda x: 1
        - x.nlargest(len(x) // N, columns="score").index.isin(x.nlargest(len(x) // N, columns="score_last").index).sum()
        / (len(x) // N)
    )
    bottom = pred.groupby(level="datetime").apply(
        lambda x: 1
        - x.nsmallest(len(x) // N, columns="score")
        .index.isin(x.nsmallest(len(x) // N, columns="score_last").index)
        .sum()
        / (len(x) // N)
    )
    r_df = pd.DataFrame(
        {
            "Top": top,
            "Bottom": bottom,
        }
    )
    turnover_figure = LinePyechartsGraph(
        r_df,
        title="Top-Bottom Turnover",
        width="1000px",
        height="500px"
    ).chart
    return (turnover_figure,)


def ic_figure_pyecharts(ic_df: pd.DataFrame, show_nature_day=True, **kwargs):
    """使用pyecharts绘制IC柱状图"""
    if show_nature_day:
        date_index = pd.date_range(ic_df.index.min(), ic_df.index.max())
        ic_df = ic_df.reindex(date_index)
    ic_bar_figure = BarPyechartsGraph(
        ic_df,
        title="Information Coefficient (IC)",
        width="1000px",
        height="500px"
    ).chart
    return ic_bar_figure


def model_performance_graph_pyecharts(
    pred_label: pd.DataFrame,
    lag: int = 1,
    N: int = 5,
    reverse=False,
    rank=False,
    graph_names: list = ["group_return", "pred_ic", "pred_autocorr"],
    show_notebook: bool = True,
    show_nature_day: bool = False,
    **kwargs,
) -> [list, tuple]:
    """
    使用pyecharts绘制模型性能图

    :param pred_label: index is **pd.MultiIndex**, index name is **[instrument, datetime]**; columns names is **[score, label]**.
           It is usually same as the label of model training(e.g. "Ref($close, -2)/Ref($close, -1) - 1").

            .. code-block:: python

                instrument  datetime        score       label
                SH600004    2017-12-11  -0.013502       -0.013502
                                2017-12-12  -0.072367       -0.072367
                                2017-12-13  -0.068605       -0.068605
                                2017-12-14  0.012440        0.012440
                                2017-12-15  -0.102778       -0.102778

    :param lag: `pred.groupby(level='instrument')['score'].shift(lag)`. It will be only used in the auto-correlation computing.
    :param N: group number, default 5.
    :param reverse: if `True`, `pred['score'] *= -1`.
    :param rank: if **True**, calculate rank ic.
    :param graph_names: graph names; default ['group_return', 'pred_ic', 'pred_autocorr', 'pred_turnover'].
    :param show_notebook: whether to display graphics in notebook, the default is `True`.
    :param show_nature_day: whether to display the abscissa of non-trading day.
    :param **kwargs: contains some parameters to control plot style.
    :return: if show_notebook is True, display in notebook; else return pyecharts chart list.
    """
    figure_list = []

    # 映射函数名到pyecharts版本
    function_mapping = {
        "group_return": _group_return_pyecharts,
        "pred_ic": _pred_ic_pyecharts,
        "pred_autocorr": _pred_autocorr_pyecharts,
        "pred_turnover": _pred_turnover_pyecharts,
    }

    for graph_name in graph_names:
        if graph_name in function_mapping:
            fun_res = function_mapping[graph_name](
                pred_label=pred_label, lag=lag, N=N, reverse=reverse, rank=rank, show_nature_day=show_nature_day, **kwargs
            )
            figure_list += fun_res

    if show_notebook:
        show_graph_in_notebook(figure_list)
    else:
        return figure_list