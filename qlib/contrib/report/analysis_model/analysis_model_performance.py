'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2026-01-18 15:30:02
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2026-01-18 16:00:05
Description: 使用pyecharts重构模型表现分析图表
'''

import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Union

# 引入 Pyecharts 组件
from pyecharts.charts import Line, Bar, Grid
from pyecharts import options as opts

# 引入我们封装好的 graph 组件 (请确保 graph.py 路径正确)
from qlib.contrib.report.graph import SubplotsGraph, ScatterGraph, BarGraph, DistplotGraph

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
            "yaxis": {"title": "Sample Quantiles"}
        },
        graph_kwargs={
            "mode": "lines",  # 用线模式模拟密集点
            "is_symbol_show": False
        }
    )
    
    if show_notebook:
        ScatterGraph.show_graph_in_notebook([graph.figure])
    return graph.figure


def _group_return(
    pred_label: pd.DataFrame, 
    N: int = 5, 
    reverse: bool = False, 
    show_notebook: bool = True
) -> object:
    """
    绘制分组累计收益图
    """
    df = pred_label.copy()
    if reverse:
        df["score"] *= -1

    # 1. 分箱
    def get_group(x):
        try:
            return pd.qcut(x, N, labels=False, duplicates='drop')
        except ValueError:
            return np.nan

    df["group"] = df.groupby("datetime")["score"].transform(get_group)
    
    # 2. 计算各组收益 (单利累加)
    group_ret = df.groupby(["datetime", "group"])["label"].mean().unstack()
    group_cum_ret = group_ret.cumsum()
    group_cum_ret.columns = [f"Group{i+1}" for i in range(len(group_cum_ret.columns))]
    
    # 3. 计算多空收益 (Long-Short)
    if not group_cum_ret.empty:
        group_cum_ret["Long-Short"] = group_cum_ret.iloc[:, -1] - group_cum_ret.iloc[:, 0]

    # 4. 绘图
    graph = ScatterGraph(
        df=group_cum_ret,
        layout={
            "title": "Cumulative Return of Groups",
            "width": "100%",
            "height": 500,
            "xaxis": {"title": "Date"},
            "yaxis": {"title": "Cumulative Return (Simple Interest)"}
        },
        graph_kwargs={"mode": "lines"}
    )
    
    if show_notebook:
        ScatterGraph.show_graph_in_notebook([graph.figure])
    return graph.figure


def _pred_ic(
    pred_label: pd.DataFrame, 
    rank: bool = False, 
    show_notebook: bool = True
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
    df_ic_ts = pd.DataFrame({
        "Daily IC": daily_ic,
        "IC MA20": daily_ic.rolling(20).mean()
    })
    
    # 3. 绘图
    # 图A: IC 时序
    graph_ts = ScatterGraph(
        df=df_ic_ts,
        layout={"title": "IC Series", "width": "100%", "height": 400},
        graph_kwargs={"mode": "lines"}
    )
    
    # 图B: IC 分布
    graph_dist = DistplotGraph(
        df=daily_ic.to_frame("IC"),
        layout={"title": "IC Distribution", "width": "100%", "height": 400}
    )
    
    figs = [graph_ts.figure, graph_dist.figure]
    
    if show_notebook:
        SubplotsGraph.show_graph_in_notebook(figs)
    return figs


def _pred_autocorr(
    pred_label: pd.DataFrame, 
    lag: int = 1, 
    show_notebook: bool = True
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
        score_unstack = score.reset_index().pivot(index="datetime", columns="instrument", values="score")
        
    for l in lags:
        # 计算每只股票的自相关系数，取平均
        ac = score_unstack.apply(lambda x: x.autocorr(lag=l)).mean()
        ac_data[f"Lag-{l}"] = ac
        
    df_ac = pd.DataFrame(list(ac_data.values()), index=list(ac_data.keys()), columns=["Autocorr"])
    
    # 2. 绘图 (柱状图)
    graph = BarGraph(
        df=df_ac,
        layout={
            "title": "Prediction Autocorrelation",
            "width": "100%", 
            "height": 400,
            "yaxis": {"title": "Autocorrelation Coefficient"}
        }
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
    show_notebook: bool = True,
) -> List[object]:
    """
    生成完整的模型性能分析报告
    
    :param pred_label: 包含 score 和 label 的 DataFrame
    :param lag: 滞后天数
    :param N: 分组数
    :param reverse: 是否反转分数
    :param rank: 是否使用 Rank IC
    :param show_notebook: 是否在 Notebook 中显示
    """
    
    # 1. 生成分组收益图
    fig_group = _group_return(pred_label, N=N, reverse=reverse, show_notebook=False)
    
    # 2. 生成 IC 分析图 (返回两个图：时序+分布)
    figs_ic = _pred_ic(pred_label, rank=rank, show_notebook=False)
    
    # 3. 生成自相关图
    fig_ac = _pred_autocorr(pred_label, lag=lag, show_notebook=False)
    
    # 4. (可选) 生成 QQ 图，如果数据量过大可注释掉
    # fig_qq = _plot_qq(pred_label["score"], show_notebook=False)
    
    # 汇总所有图表对象
    all_figs = [fig_group] + figs_ic + [fig_ac]
    
    # 统一显示
    if show_notebook:
        SubplotsGraph.show_graph_in_notebook(all_figs)
        
    return all_figs