'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2026-01-17 00:35:10
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2026-01-17 00:36:34
Description: pyecharts重构
'''
import pandas as pd
from pyecharts import options as opts

# 引入我们在 Step 2 和 Step 3 重构的 graph 模块
from ..graph import SubplotsGraph, BaseGraph


def _calculate_maximum(df: pd.DataFrame, is_ex: bool = False):
    """
    计算最大回撤区间 (保持原有逻辑不变)
    """
    if is_ex:
        end_date = df["cum_ex_return_wo_cost_mdd"].idxmin()
        start_date = df.loc[df.index <= end_date]["cum_ex_return_wo_cost"].idxmax()
    else:
        end_date = df["return_wo_mdd"].idxmin()
        start_date = df.loc[df.index <= end_date]["cum_return_wo_cost"].idxmax()
    return start_date, end_date


def _calculate_mdd(series):
    """
    计算回撤序列 (保持原有逻辑不变)
    """
    return series - series.cummax()


def _calculate_report_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    准备绘图数据 (保持原有逻辑不变)
    """
    index_names = df.index.names
    # 格式化日期索引以便于显示
    df.index = df.index.strftime("%Y-%m-%d")

    report_df = pd.DataFrame()

    report_df["cum_bench"] = df["bench"].cumsum()
    report_df["cum_return_wo_cost"] = df["return"].cumsum()
    report_df["cum_return_w_cost"] = (df["return"] - df["cost"]).cumsum()
    
    report_df["return_wo_mdd"] = _calculate_mdd(report_df["cum_return_wo_cost"])
    report_df["return_w_cost_mdd"] = _calculate_mdd((df["return"] - df["cost"]).cumsum())

    report_df["cum_ex_return_wo_cost"] = (df["return"] - df["bench"]).cumsum()
    report_df["cum_ex_return_w_cost"] = (df["return"] - df["bench"] - df["cost"]).cumsum()
    report_df["cum_ex_return_wo_cost_mdd"] = _calculate_mdd((df["return"] - df["bench"]).cumsum())
    report_df["cum_ex_return_w_cost_mdd"] = _calculate_mdd((df["return"] - df["cost"] - df["bench"]).cumsum())

    report_df["turnover"] = df["turnover"]
    report_df.sort_index(ascending=True, inplace=True)

    report_df.index.names = index_names
    return report_df


def _report_figure(df: pd.DataFrame) -> list:
    # 1. 获取并处理数据
    report_df = _calculate_report_data(df)
    max_start_date, max_end_date = _calculate_maximum(report_df)
    ex_max_start_date, ex_max_end_date = _calculate_maximum(report_df, True)

    index_name = report_df.index.name
    _temp_df = report_df.reset_index()
    _temp_df.loc[-1] = 0
    _temp_df = _temp_df.shift(1)
    _temp_df.loc[0, index_name] = "Start" 
    _temp_df.set_index(index_name, inplace=True)
    _temp_df.iloc[0] = 0
    report_df = _temp_df

    # 2. 构造 MarkArea (最大回撤阴影)
    mark_area_items = []
    if max_start_date and max_end_date:
        mark_area_items.append(
            opts.MarkAreaItem(
                name="Max Drawdown",
                x=(str(max_start_date), str(max_end_date)),
                itemstyle_opts=opts.ItemStyleOpts(color="#d3d3d3", opacity=0.3)
            )
        )
    # 也可以加上超额收益回撤 (如果需要在特定图上显示不同颜色，需分开定义，这里统一显示绝对回撤)
    # 如果你想把两种回撤都画在所有图上，可能会有点乱。通常 "最大回撤" 指的是绝对收益的回撤。
    
    markarea_opts = opts.MarkAreaOpts(data=mark_area_items, is_silent=True)

    # 3. 定义图表配置
    _default_kind_map = dict(kind="ScatterGraph", kwargs={"mode": "lines+markers"})
    _temp_fill_args = {"fill": "tozeroy", "mode": "lines+markers"}
    
    # 原始配置列表
    raw_config = [
        ("cum_bench", dict(row=1, col=1)),
        ("cum_return_wo_cost", dict(row=1, col=1)),
        ("cum_return_w_cost", dict(row=1, col=1)),
        
        ("return_wo_mdd", dict(row=2, col=1, graph_kwargs=_temp_fill_args)),
        ("return_w_cost_mdd", dict(row=3, col=1, graph_kwargs=_temp_fill_args)),
        
        ("cum_ex_return_wo_cost", dict(row=4, col=1)),
        ("cum_ex_return_w_cost", dict(row=4, col=1)),
        
        ("turnover", dict(row=5, col=1)),
        
        ("cum_ex_return_w_cost_mdd", dict(row=6, col=1, graph_kwargs=_temp_fill_args)),
        ("cum_ex_return_wo_cost_mdd", dict(row=7, col=1, graph_kwargs=_temp_fill_args)),
    ]

    # --- 关键修改：为每一行注入 markarea_opts ---
    _column_row_col_dict = []
    
    # 记录哪些行已经添加过阴影了 (避免一行多列时重复添加)
    rows_with_markarea = set()

    for col_name, config in raw_config:
        row = config['row']
        
        # 获取或初始化 graph_kwargs
        if 'graph_kwargs' not in config:
            config['graph_kwargs'] = {}
            
        # 如果这一行还没加过 markarea，则加到当前的列配置中
        if row not in rows_with_markarea:
            # 浅拷贝 graph_kwargs 避免影响其他共用 _temp_fill_args 的项
            new_kwargs = config['graph_kwargs'].copy()
            new_kwargs['markarea_opts'] = markarea_opts
            config['graph_kwargs'] = new_kwargs
            rows_with_markarea.add(row)
        else:
            # 必须确保 graph_kwargs 也是独立的，防止引用污染
            config['graph_kwargs'] = config['graph_kwargs'].copy()
            
        _column_row_col_dict.append((col_name, config))


    _layout_style = dict(
        height=1200, 
        width="100%",
        title="Backtest Analysis Report",
    )

    _subplot_kwargs = dict(
        rows=7,
        cols=1,
        row_width=[1, 1, 1, 3, 1, 1, 3], # Bottom-to-Top
        vertical_spacing=0.01,
        shared_xaxes=True,
    )

    grid = SubplotsGraph(
        df=report_df,
        layout=_layout_style,
        sub_graph_data=_column_row_col_dict,
        subplots_kwargs=_subplot_kwargs,
        kind_map=_default_kind_map,
    ).figure
    
    return [grid]


def report_graph(report_df: pd.DataFrame, show_notebook: bool = True) -> list:
    report_df = report_df.copy()
    fig_list = _report_figure(report_df)
    if show_notebook:
        BaseGraph.show_graph_in_notebook(fig_list)
    else:
        return fig_list