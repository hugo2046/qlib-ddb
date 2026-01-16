#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 risk_analysis_pyecharts 组件的功能
"""
import sys
import os
import pandas as pd
import numpy as np

# 添加qlib路径
sys.path.insert(0, "/data1/hugo/workspace/qlib_ddb")


def create_test_data():
    """创建测试数据"""
    # 创建日期索引
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    # 过滤交易日（简单模拟）
    dates = dates[dates.weekday < 5]  # 周一到周五

    # 创建报告数据
    np.random.seed(42)
    n = len(dates)

    report_normal_df = pd.DataFrame({
        'return': np.random.normal(0.001, 0.02, n),
        'bench': np.random.normal(0.0008, 0.015, n),
        'cost': np.random.uniform(0.0001, 0.002, n),
        'turnover': np.random.uniform(0.1, 0.8, n)
    }, index=dates)

    # 创建分析数据
    analysis_data = {
        ('excess_return_without_cost', 'mean'): 0.000692,
        ('excess_return_without_cost', 'std'): 0.005374,
        ('excess_return_without_cost', 'annualized_return'): 0.174495,
        ('excess_return_without_cost', 'information_ratio'): 2.045576,
        ('excess_return_without_cost', 'max_drawdown'): -0.079103,
        ('excess_return_with_cost', 'mean'): 0.000499,
        ('excess_return_with_cost', 'std'): 0.005372,
        ('excess_return_with_cost', 'annualized_return'): 0.125625,
        ('excess_return_with_cost', 'information_ratio'): 1.473152,
        ('excess_return_with_cost', 'max_drawdown'): -0.088263,
    }

    analysis_df = pd.Series(analysis_data).unstack()
    analysis_df.index.name = 'strategy'
    analysis_df.columns.name = 'risk'

    return report_normal_df, analysis_df


def test_risk_analysis_pyecharts_basic():
    """测试 risk_analysis_pyecharts 基本功能"""
    print("创建测试数据...")
    report_normal_df, analysis_df = create_test_data()

    print(f"报告数据形状: {report_normal_df.shape}")
    print(f"分析数据形状: {analysis_df.shape}")

    try:
        from qlib.contrib.report.analysis_position.risk_analysis_pyecharts import (
            risk_analysis_graph_pyecharts,
            _get_risk_analysis_pyecharts_figure,
            _get_monthly_risk_analysis_pyecharts_figure
        )

        print("\n测试风险分析图表生成...")
        charts = _get_risk_analysis_pyecharts_figure(analysis_df)
        print(f"生成了 {len(charts)} 个风险分析图表")

        print("\n测试月度风险分析图表生成...")
        monthly_charts = _get_monthly_risk_analysis_pyecharts_figure(report_normal_df)
        print(f"生成了 {len(monthly_charts)} 个月度分析图表")

        print("\n测试完整图表生成...")
        # 测试不显示，只返回图表对象
        result = risk_analysis_graph_pyecharts(
            analysis_df=analysis_df,
            report_normal_df=report_normal_df,
            show_notebook=False
        )
        print(f"完整图表生成成功，返回了 {len(result) if result else 0} 个图表")

        # 测试保存到文件
        output_dir = os.path.join(os.path.dirname(__file__), "test_output")
        os.makedirs(output_dir, exist_ok=True)

        save_path = os.path.join(output_dir, "test_risk_analysis.html")
        risk_analysis_graph_pyecharts(
            analysis_df=analysis_df,
            report_normal_df=report_normal_df,
            show_notebook=False,
            save_path=save_path
        )
        print(f"图表已保存到 {save_path}")

    except ImportError as e:
        print(f"导入失败: {e}")
        raise e
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        raise e


def test_compatibility():
    """测试兼容性"""
    print("\n测试兼容性...")

    try:
        from qlib.contrib.report.analysis_position.risk_analysis import risk_analysis_graph
        print("原始 risk_analysis_graph 函数可以正常导入")
    except ImportError as e:
        print(f"原始 risk_analysis_graph 函数导入失败: {e}")

    try:
        from qlib.contrib.report.analysis_position.risk_analysis_pyecharts import risk_analysis_graph_pyecharts
        print("pyecharts 版本函数可以正常导入")
    except ImportError as e:
        print(f"pyecharts 版本函数导入失败: {e}")


def test_data_processing():
    """测试数据处理函数"""
    print("\n测试数据处理函数...")

    try:
        from qlib.contrib.report.analysis_position.risk_analysis_pyecharts import (
            _get_all_risk_analysis,
            _get_monthly_risk_analysis_with_report,
            _get_monthly_analysis_with_feature
        )

        # 创建测试数据
        report_normal_df, analysis_df = create_test_data()

        # 测试 _get_all_risk_analysis
        processed_df = _get_all_risk_analysis(analysis_df)
        print(f"_get_all_risk_analysis 输出形状: {processed_df.shape}")
        print(f"处理后的数据列: {processed_df.columns.tolist()}")

        # 测试 _get_monthly_risk_analysis_with_report
        monthly_df = _get_monthly_risk_analysis_with_report(report_normal_df)
        print(f"_get_monthly_risk_analysis_with_report 输出形状: {monthly_df.shape}")

        # 测试 _get_monthly_analysis_with_feature
        if not monthly_df.empty:
            feature_df = _get_monthly_analysis_with_feature(monthly_df, "annualized_return")
            print(f"_get_monthly_analysis_with_feature 输出形状: {feature_df.shape}")

        print("数据处理函数测试完成!")

    except ImportError as e:
        print(f"导入数据处理函数失败: {e}")
    except Exception as e:
        print(f"数据处理函数测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def test_edge_cases():
    """测试边界情况"""
    print("\n测试边界情况...")

    try:
        from qlib.contrib.report.analysis_position.risk_analysis_pyecharts import (
            risk_analysis_graph_pyecharts,
            _get_risk_analysis_pyecharts_figure,
            _get_monthly_risk_analysis_pyecharts_figure
        )

        # 测试空数据
        print("测试空数据...")
        empty_df = pd.DataFrame()
        empty_charts = _get_risk_analysis_pyecharts_figure(empty_df)
        print(f"空数据生成了 {len(empty_charts)} 个图表")

        # 测试 None 数据
        print("测试 None 数据...")
        none_charts = _get_risk_analysis_pyecharts_figure(None)
        print(f"None 数据生成了 {len(none_charts)} 个图表")

        # 测试最小数据
        print("测试最小数据...")
        minimal_report = pd.DataFrame({
            'return': [0.01, 0.02, 0.015],
            'bench': [0.008, 0.01, 0.009],
            'cost': [0.001, 0.001, 0.001],
            'turnover': [0.5, 0.6, 0.4]
        }, index=pd.date_range('2020-01-01', periods=3))

        minimal_charts = _get_monthly_risk_analysis_pyecharts_figure(minimal_report)
        print(f"最小数据生成了 {len(minimal_charts)} 个月度图表")

        print("边界情况测试完成!")

    except ImportError as e:
        print(f"导入失败: {e}")
    except Exception as e:
        print(f"边界情况测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 50)
    print("测试 risk_analysis_pyecharts 模块")
    print("=" * 50)

    test_compatibility()
    test_data_processing()
    test_risk_analysis_pyecharts_basic()
    test_edge_cases()

    print("\n" + "=" * 50)
    print("所有测试完成")
    print("=" * 50)