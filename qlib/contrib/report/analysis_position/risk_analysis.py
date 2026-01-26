"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2025-02-18 11:26:04
LastEditors: shen.lan123@gmail.com
LastEditTime: 2026-01-19 13:00:45
Description: pyecharts重构风险分析图
"""

from typing import Iterable, Any

import pandas as pd
from pyecharts.commons.utils import JsCode  # [New] 支持 JS Formatter

from ...evaluate import risk_analysis
from ..graph import (
    SubplotsGraph,
    ScatterGraph,
    get_percent_formatter,
    get_axis_percent_formatter,
    get_number_formatter,  # [New]
)  # [New] 导入工具函数
from ..display_config import RISK_ANALYSIS_SUBPLOTS_CONFIG, MONTHLY_RISK_SUBPLOTS_CONFIG


def _get_risk_analysis_data_with_report(
    report_normal_df: pd.DataFrame,
    # report_long_short_df: pd.DataFrame,
    date: pd.Timestamp,
) -> pd.DataFrame:
    """Get risk analysis data with report

    :param report_normal_df: report data
    :param report_long_short_df: report data
    :param date: date string
    :return:
    """

    analysis = dict()
    # if not report_long_short_df.empty:
    #     analysis["pred_long"] = risk_analysis(report_long_short_df["long"])
    #     analysis["pred_short"] = risk_analysis(report_long_short_df["short"])
    #     analysis["pred_long_short"] = risk_analysis(report_long_short_df["long_short"])

    if not report_normal_df.empty:
        analysis["excess_return_without_cost"] = risk_analysis(
            report_normal_df["return"] - report_normal_df["bench"]
        )
        analysis["excess_return_with_cost"] = risk_analysis(
            report_normal_df["return"]
            - report_normal_df["bench"]
            - report_normal_df["cost"]
        )
    analysis_df = pd.concat(analysis)  # type: pd.DataFrame
    analysis_df["date"] = date
    return analysis_df


def _get_all_risk_analysis(risk_df: pd.DataFrame) -> pd.DataFrame:
    """risk_df to standard

    :param risk_df: risk data
    :return:
    """
    if risk_df is None:
        return pd.DataFrame()
    risk_df = risk_df.unstack()
    risk_df.columns = risk_df.columns.droplevel(0)
    return risk_df.drop("mean", axis=1)


def _get_monthly_risk_analysis_with_report(
    report_normal_df: pd.DataFrame,
) -> pd.DataFrame:
    """Get monthly analysis data

    :param report_normal_df:
    # :param report_long_short_df:
    :return:
    """

    # Group by month
    report_normal_gp = report_normal_df.groupby(
        [report_normal_df.index.year, report_normal_df.index.month]
    )
    # report_long_short_gp = report_long_short_df.groupby(
    #     [report_long_short_df.index.year, report_long_short_df.index.month]
    # )

    gp_month = sorted(set(report_normal_gp.size().index))

    _monthly_df = pd.DataFrame()
    for gp_m in gp_month:
        _m_report_normal = report_normal_gp.get_group(gp_m)
        # _m_report_long_short = report_long_short_gp.get_group(gp_m)

        if len(_m_report_normal) < 3:
            # The month's data is less than 3, not displayed
            # FIXME: If the trading day of a month is less than 3 days, a breakpoint will appear in the graph
            continue
        month_days = pd.Timestamp(year=gp_m[0], month=gp_m[1], day=1).days_in_month
        _temp_df = _get_risk_analysis_data_with_report(
            _m_report_normal,
            # _m_report_long_short,
            pd.Timestamp(year=gp_m[0], month=gp_m[1], day=month_days),
        )
        _monthly_df = pd.concat([_monthly_df, _temp_df], sort=False)

    return _monthly_df


def _get_monthly_analysis_with_feature(
    monthly_df: pd.DataFrame, feature: str = "annualized_return"
) -> pd.DataFrame:
    """

    :param monthly_df:
    :param feature:
    :return:
    """
    # Use scalar grouping key to avoid pandas warning about length-1 list-like keys
    _monthly_df_gp = monthly_df.reset_index().groupby("level_1")

    _name_df = _monthly_df_gp.get_group(feature).set_index(["level_0", "level_1"])
    _temp_df = _name_df.pivot_table(
        index="date", values=["risk"], columns=_name_df.index
    )
    _temp_df.columns = map(lambda x: "_".join(x[-1]), _temp_df.columns)
    _temp_df.index = _temp_df.index.strftime("%Y-%m")

    return _temp_df


def _get_risk_analysis_figure(analysis_df: pd.DataFrame) -> Iterable[Any]:
    """Get analysis graph figure(Bar Chart)

    :param analysis_df:
    :return:
    """
    if analysis_df is None:
        return []

    # _figure = SubplotsGraph(
    #     _get_all_risk_analysis(analysis_df),
    #     kind_map=dict(kind="BarGraph", kwargs={}),
    #     subplots_kwargs={"rows": 1, "cols": 4},
    # ).figure

    # use echarts
    _figure = SubplotsGraph(
        _get_all_risk_analysis(analysis_df),
        config=RISK_ANALYSIS_SUBPLOTS_CONFIG,
    ).figure
    return (_figure,)


def _get_monthly_risk_analysis_figure(report_normal_df: pd.DataFrame) -> Iterable[Any]:
    """Get analysis monthly graph figure(Line Chart)

    :param report_normal_df:
    :param report_long_short_df:
    :return:
    """

    if report_normal_df is None:
        return []

    _monthly_df: pd.DataFrame = _get_monthly_risk_analysis_with_report(
        report_normal_df=report_normal_df,
    )

    # 转为宽表格式，index-date,columns-(metric_feature, risk)
    df: pd.DataFrame = (
        _monthly_df.set_index("date", append=True)["risk"]
        .unstack(level=[0, 1])
        .swaplevel(0, 1, axis=1)
    )

    # 准备容器
    processed_dfs = []
    sub_graph_data = []

    base_kwargs = {
        "mode": "lines+markers",
        "legend_pos_left": None,
        "legend_pos_right": "5%",
        "legend_pos_top": None,  # 让 SubplotsGraph 自动计算
        "title_top_offset": -4,  # 标题上移
        "is_show_legend": True,
    }

    # feature为mean,std,annualized_return,information_ratio
    for i, (feature, df_grp) in enumerate(df.groupby(axis=1, level=0)):
        row_idx = i + 1
        # 此时index-date,columns-excess_return_without_code和excess_return_with_code
        sub_df = df_grp[feature].copy()
        current_kwargs = base_kwargs.copy()

        # [Modify] 核心修改点：使用 JS Formatter
        if feature == "information_ratio":
            current_kwargs["axis_formatter"] = JsCode(get_number_formatter(2))
            current_kwargs["tooltip_formatter"] = None  # IR 不需要百分比格式化
            # unit_suffix = ""
        else:
            # 1. 移除 Python 端的 multiply(100)，保留原始小数精度
            # sub_df = sub_df.multiply(100).round(4)  # <-- REMOVED

            # 2. Tooltip 使用 JS Formatter (自动乘100并保留4位小数)
            current_kwargs["tooltip_formatter"] = JsCode(get_percent_formatter(2))

            # 3. Y轴也需要改为 JS Formatter (否则会显示 0.24 %)
            current_kwargs["axis_formatter"] = JsCode(get_axis_percent_formatter(2))

            # unit_suffix = " (%)"

        rename_map = {col: f"{feature}:{col}" for col in sub_df.columns}
        sub_df.rename(columns=rename_map, inplace=True)

        # 收集数据
        processed_dfs.append(sub_df)

        for unique_col in sub_df.columns:
            original_source_name = unique_col.split(":")[-1]
            display_name = original_source_name.replace("_", " ").title()
            # display_name += unit_suffix
            sub_graph_data.append(
                (
                    unique_col,
                    dict(
                        row=row_idx,
                        col=1,
                        title=feature,
                        name=display_name,  # 图例显示内容
                        kind="ScatterGraph",
                        graph_kwargs=current_kwargs,
                    ),
                )
            )

        # 4. 合并数据并绘图
    if not processed_dfs:
        return []

    final_df = pd.concat(processed_dfs, axis=1)

    # [Change] 使用 config 并仅覆盖动态参数
    _figure = SubplotsGraph(
        final_df,
        sub_graph_data=sub_graph_data,
        config=MONTHLY_RISK_SUBPLOTS_CONFIG,
        subplots_kwargs={
            "rows": len(processed_dfs),
            "row_width": [1] * len(processed_dfs),
            "vertical_spacing": 0.05,
        },
    ).figure

    return (_figure,)

    # for _feature in ["annualized_return", "max_drawdown", "information_ratio", "std"]:
    #     _temp_df = _get_monthly_analysis_with_feature(_monthly_df, _feature)
    #     yield ScatterGraph(
    #         _temp_df,
    #         # layout=dict(title=_feature, xaxis=dict(type="category", tickangle=45)),
    #         graph_kwargs=graph_kwargs,
    #         layout={"title": _feature, "title_pos_left": "center"},
    #     ).figure


def risk_analysis_graph(
    analysis_df: pd.DataFrame = None,
    report_normal_df: pd.DataFrame = None,
    report_long_short_df: pd.DataFrame = None,
    show_notebook: bool = True,
) -> Iterable[Any]:
    """Generate analysis graph and monthly analysis

        Example:


            .. code-block:: python

                import qlib
                import pandas as pd
                from qlib.utils.time import Freq
                from qlib.utils import flatten_dict
                from qlib.backtest import backtest, executor
                from qlib.contrib.evaluate import risk_analysis
                from qlib.contrib.strategy import TopkDropoutStrategy

                # init qlib
                qlib.init(provider_uri=<qlib data dir>)

                CSI300_BENCH = "SH000300"
                FREQ = "day"
                STRATEGY_CONFIG = {
                    "topk": 50,
                    "n_drop": 5,
                    # pred_score, pd.Series
                    "signal": pred_score,
                }

                EXECUTOR_CONFIG = {
                    "time_per_step": "day",
                    "generate_portfolio_metrics": True,
                }

                backtest_config = {
                    "start_time": "2017-01-01",
                    "end_time": "2020-08-01",
                    "account": 100000000,
                    "benchmark": CSI300_BENCH,
                    "exchange_kwargs": {
                        "freq": FREQ,
                        "limit_threshold": 0.095,
                        "deal_price": "close",
                        "open_cost": 0.0005,
                        "close_cost": 0.0015,
                        "min_cost": 5,
                    },
                }

                # strategy object
                strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
                # executor object
                executor_obj = executor.SimulatorExecutor(**EXECUTOR_CONFIG)
                # backtest
                portfolio_metric_dict, indicator_dict = backtest(executor=executor_obj, strategy=strategy_obj, **backtest_config)
                analysis_freq = "{0}{1}".format(*Freq.parse(FREQ))
                # backtest info
                report_normal_df, positions_normal = portfolio_metric_dict.get(analysis_freq)
                analysis = dict()
                analysis["excess_return_without_cost"] = risk_analysis(
                    report_normal_df["return"] - report_normal_df["bench"], freq=analysis_freq
                )
                analysis["excess_return_with_cost"] = risk_analysis(
                    report_normal_df["return"] - report_normal_df["bench"] - report_normal_df["cost"], freq=analysis_freq
                )

                analysis_df = pd.concat(analysis)  # type: pd.DataFrame
                analysis_position.risk_analysis_graph(analysis_df, report_normal_df)



    :param analysis_df: analysis data, index is **pd.MultiIndex**; columns names is **[risk]**.


            .. code-block:: python

                                                                  risk
                excess_return_without_cost mean               0.000692
                                           std                0.005374
                                           annualized_return  0.174495
                                           information_ratio  2.045576
                                           max_drawdown      -0.079103
                excess_return_with_cost    mean               0.000499
                                           std                0.005372
                                           annualized_return  0.125625
                                           information_ratio  1.473152
                                           max_drawdown      -0.088263


    :param report_normal_df: **df.index.name** must be **date**, df.columns must contain **return**, **turnover**, **cost**, **bench**.


            .. code-block:: python

                            return      cost        bench       turnover
                date
                2017-01-04  0.003421    0.000864    0.011693    0.576325
                2017-01-05  0.000508    0.000447    0.000721    0.227882
                2017-01-06  -0.003321   0.000212    -0.004322   0.102765
                2017-01-09  0.006753    0.000212    0.006874    0.105864
                2017-01-10  -0.000416   0.000440    -0.003350   0.208396


    :param report_long_short_df: **df.index.name** must be **date**, df.columns contain **long**, **short**, **long_short**.


            .. code-block:: python

                            long        short       long_short
                date
                2017-01-04  -0.001360   0.001394    0.000034
                2017-01-05  0.002456    0.000058    0.002514
                2017-01-06  0.000120    0.002739    0.002859
                2017-01-09  0.001436    0.001838    0.003273
                2017-01-10  0.000824    -0.001944   -0.001120


    :param show_notebook: Whether to display graphics in a notebook, default **True**.
        If True, show graph in notebook
        If False, return graph figure
    :return:
    """
    _figure_list = list(_get_risk_analysis_figure(analysis_df)) + list(
        _get_monthly_risk_analysis_figure(
            report_normal_df,
            # report_long_short_df,
        )
    )
    if show_notebook:
        ScatterGraph.show_graph_in_notebook(_figure_list)
    else:
        return _figure_list
