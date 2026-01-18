'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2026-01-17 00:35:10
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2026-01-18 20:46:48
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
        start_date = df.loc[df.index <=
                            end_date]["cum_ex_return_wo_cost"].idxmax()
    else:
        end_date = df["return_wo_mdd"].idxmin()
        start_date = df.loc[df.index <=
                            end_date]["cum_return_wo_cost"].idxmax()
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

    report_df["return_wo_mdd"] = _calculate_mdd(
        report_df["cum_return_wo_cost"])
    report_df["return_w_cost_mdd"] = _calculate_mdd(
        (df["return"] - df["cost"]).cumsum())

    report_df["cum_ex_return_wo_cost"] = (df["return"] - df["bench"]).cumsum()
    report_df["cum_ex_return_w_cost"] = (
        df["return"] - df["bench"] - df["cost"]).cumsum()
    report_df["cum_ex_return_wo_cost_mdd"] = _calculate_mdd(
        (df["return"] - df["bench"]).cumsum())
    report_df["cum_ex_return_w_cost_mdd"] = _calculate_mdd(
        (df["return"] - df["cost"] - df["bench"]).cumsum())

    report_df["turnover"] = df["turnover"]
    report_df.sort_index(ascending=True, inplace=True)

    report_df.index.names = index_names
    return report_df


def _report_figure(df: pd.DataFrame) -> list:
    # 1. 获取并处理数据
    report_df = _calculate_report_data(df)
    max_start_date, max_end_date = _calculate_maximum(report_df, is_ex=False)
    ex_max_start_date, ex_max_end_date = _calculate_maximum(report_df, is_ex=True)

    index_name = report_df.index.name
    _temp_df = report_df.reset_index()
    _temp_df.loc[-1] = 0
    _temp_df = _temp_df.shift(1)
    _temp_df.loc[0, index_name] = "Start"
    _temp_df.set_index(index_name, inplace=True)
    _temp_df.iloc[0] = 0
    
    # 手动转换数据为百分比数值 (适配 graph.py 的纯净模式)
    report_df = _temp_df * 100

    # 2. 构造 MarkArea (回撤阴影)
    
    # 2.1 绝对收益最大回撤 (用于图 1, 2, 3)
    mark_area_items_normal = []
    if max_start_date and max_end_date:
        mark_area_items_normal.append(
            opts.MarkAreaItem(
                name="Max Drawdown",
                x=(str(max_start_date), str(max_end_date)),
                itemstyle_opts=opts.ItemStyleOpts(color="#d3d3d3", opacity=0.3)
            )
        )
    markarea_opts_normal = opts.MarkAreaOpts(data=mark_area_items_normal, is_silent=True)

    # 2.2 超额收益最大回撤 (用于图 4, 5, 6, 7)
    mark_area_items_excess = []
    if ex_max_start_date and ex_max_end_date:
        mark_area_items_excess.append(
            opts.MarkAreaItem(
                name="Excess Max Drawdown",
                x=(str(ex_max_start_date), str(ex_max_end_date)),
                # 可以选择不同颜色区分，或者保持一致
                itemstyle_opts=opts.ItemStyleOpts(color="#d3d3d3", opacity=0.3) 
            )
        )
    markarea_opts_excess = opts.MarkAreaOpts(data=mark_area_items_excess, is_silent=True)

    # 3. 定义图表配置
    _default_kind_map = dict(kind="ScatterGraph",
                             kwargs={"mode": "lines+markers",
                                     "fill": "tozeroy",
                                     "axis_formatter": "{value} %",
                                     "legend_pos_left": None,
                                     "legend_pos_right": "5%"})
    
    _temp_fill_args = {"fill": "tozeroy", 
                       "mode": "lines+markers",
                       "axis_formatter": "{value} %", 
                       "legend_pos_left": None,
                       "legend_pos_right": "5%" }

    # 原始配置列表
    raw_config = [
        # --- Figure 1 (Absolute Return) ---
        ("cum_bench", dict(row=1, col=1)),
        ("cum_return_wo_cost", dict(row=1, col=1)),
        ("cum_return_w_cost", dict(row=1, col=1)),

        # --- Figure 2, 3 (Absolute Drawdown) ---
        ("return_wo_mdd", dict(row=2, col=1, graph_kwargs=_temp_fill_args)),
        ("return_w_cost_mdd", dict(row=3, col=1, graph_kwargs=_temp_fill_args)),

        # --- Figure 4 (Excess Return) ---
        ("cum_ex_return_wo_cost", dict(row=4, col=1)),
        ("cum_ex_return_w_cost", dict(row=4, col=1)),

        # --- Figure 5 (Turnover) ---
        ("turnover", dict(row=5, col=1)),

        # --- Figure 6, 7 (Excess Drawdown) ---
        ("cum_ex_return_w_cost_mdd", dict(
            row=6, col=1, graph_kwargs=_temp_fill_args)),
        ("cum_ex_return_wo_cost_mdd", dict(
            row=7, col=1, graph_kwargs=_temp_fill_args)),
    ]

    _column_row_col_dict = []

    rows_with_markarea = set()

    for col_name, config in raw_config:
        row = config['row']

        if 'graph_kwargs' not in config:
            config['graph_kwargs'] = {}

        # [Fix] 核心修改：根据行号分配回撤阴影
        current_markarea = None
        
        if row in [1, 2, 3]:
            # 图 1, 2, 3 使用绝对收益回撤
            current_markarea = markarea_opts_normal
        elif row in [4, 5, 6, 7]:
            # 图 4, 5, 6, 7 使用超额收益回撤
            current_markarea = markarea_opts_excess
        
        # 仅为每行的第一个图表添加配置，避免重复
        if row not in rows_with_markarea:
            new_kwargs = config['graph_kwargs'].copy()
            if current_markarea:
                new_kwargs['markarea_opts'] = current_markarea
            config['graph_kwargs'] = new_kwargs
            rows_with_markarea.add(row)
        else:
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
        row_width=[1, 1, 1, 3, 1, 1, 3],  # Bottom-to-Top
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
