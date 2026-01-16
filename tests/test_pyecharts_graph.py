#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试pyecharts_graph组件的功能
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加qlib路径
sys.path.insert(0, "/data1/hugo/workspace/qlib_ddb")

def create_test_data():
    """创建测试数据"""
    # 创建日期索引
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')

    # 过滤交易日（排除周末）
    dates = dates[dates.weekday < 5]

    # 创建股票代码
    instruments = ['SH600000', 'SH600001', 'SH600002', 'SH600003', 'SH600004']

    # 创建MultiIndex
    index = pd.MultiIndex.from_product([instruments, dates], names=['instrument', 'datetime'])

    # 生成随机分数和标签
    np.random.seed(42)
    n = len(index)

    # 生成具有相关性的分数和标签
    score = np.random.normal(0, 1, n)
    # 添加一些趋势
    trend = np.linspace(0, 0.5, n)
    score = score + trend

    # 生成标签，与分数有一定相关性
    noise = np.random.normal(0, 0.5, n)
    label = score * 0.3 + noise

    # 创建DataFrame
    df = pd.DataFrame({
        'score': score,
        'label': label
    }, index=index)

    return df

def test_pyecharts_graph():
    """测试pyecharts图形功能"""
    print("创建测试数据...")
    pred_label = create_test_data()
    print(f"测试数据形状: {pred_label.shape}")
    print(f"数据示例:\n{pred_label.head()}")

    try:
        # 导入pyecharts版本的分析函数
        from qlib.contrib.report.analysis_model.analysis_model_performance_pyecharts import (
            model_performance_graph_pyecharts
        )

        print("\n测试pyecharts版本的模型性能分析...")

        # 测试pyecharts版本
        figures = model_performance_graph_pyecharts(
            pred_label=pred_label,
            graph_names=["group_return", "pred_ic", "pred_autocorr"],
            show_notebook=False,
            N=5,
            show_nature_day=False
        )

        print(f"生成了 {len(figures)} 个图形")

        # 保存图形到文件
        output_dir = os.path.join(os.path.dirname(__file__), "test_output")
        os.makedirs(output_dir, exist_ok=True)

        for i, fig in enumerate(figures):
            if hasattr(fig, 'render'):
                # 如果是pyecharts图表
                file_path = os.path.join(output_dir, f"chart_{i}.html")
                fig.render(file_path)
                print(f"图形 {i} 已保存到: {file_path}")
            elif hasattr(fig, 'charts'):
                # 如果是SubplotsPyechartsGraph
                file_path = os.path.join(output_dir, f"subplots_{i}.html")
                fig.render(file_path)
                print(f"子图 {i} 已保存到: {file_path}")

        print("pyecharts版本测试完成!")

    except ImportError as e:
        print(f"导入pyecharts版本失败: {e}")
        print("请确保安装了pyecharts: pip install pyecharts")
        raise e
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def test_compatibility():
    """测试兼容性"""
    print("\n测试兼容性...")

    pred_label = create_test_data()

    try:
        from qlib.contrib.report.analysis_model.analysis_model_performance import model_performance_graph

        # 测试plotly版本（默认）
        print("测试plotly版本...")
        figures_plotly = model_performance_graph(
            pred_label=pred_label,
            graph_names=["group_return", "pred_ic"],
            show_notebook=False,
            use_pyecharts=False
        )
        print(f"plotly版本生成了 {len(figures_plotly)} 个图形")

        # 测试pyecharts版本
        print("测试pyecharts版本...")
        figures_pyecharts = model_performance_graph(
            pred_label=pred_label,
            graph_names=["group_return", "pred_ic"],
            show_notebook=False,
            use_pyecharts=True
        )
        print(f"pyecharts版本生成了 {len(figures_pyecharts)} 个图形")

        print("兼容性测试完成!")

    except ImportError as e:
        print(f"导入模块失败: {e}")
    except Exception as e:
        print(f"兼容性测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def test_individual_components():
    """测试单个组件"""
    print("\n测试单个pyecharts组件...")

    try:
        from qlib.contrib.report.pyecharts_graph import (
            ScatterPyechartsGraph, BarPyechartsGraph, LinePyechartsGraph,
            HeatmapPyechartsGraph, HistogramPyechartsGraph, SubplotsPyechartsGraph
        )

        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Series1': np.random.randn(100).cumsum(),
            'Series2': np.random.randn(100).cumsum(),
            'Series3': np.random.randn(100).cumsum()
        }, index=dates)

        # 测试散点图
        print("测试散点图...")
        scatter_chart = ScatterPyechartsGraph(data, title="测试散点图")
        scatter_chart.render(os.path.join(os.path.dirname(__file__), "test_output", "scatter_test.html"))
        print("散点图测试完成")

        # 测试折线图
        print("测试折线图...")
        line_chart = LinePyechartsGraph(data, title="测试折线图")
        line_chart.render(os.path.join(os.path.dirname(__file__), "test_output", "line_test.html"))
        print("折线图测试完成")

        # 测试柱状图
        print("测试柱状图...")
        bar_data = data.iloc[-30:]  # 取最后30天
        bar_chart = BarPyechartsGraph(bar_data, title="测试柱状图")
        bar_chart.render(os.path.join(os.path.dirname(__file__), "test_output", "bar_test.html"))
        print("柱状图测试完成")

        # 测试热力图
        print("测试热力图...")
        heatmap_data = pd.DataFrame(
            np.random.randn(10, 8),
            index=[f"Row_{i}" for i in range(10)],
            columns=[f"Col_{j}" for j in range(8)]
        )
        heatmap_chart = HeatmapPyechartsGraph(heatmap_data, title="测试热力图")
        heatmap_chart.render(os.path.join(os.path.dirname(__file__), "test_output", "heatmap_test.html"))
        print("热力图测试完成")

        # 测试直方图
        print("测试直方图...")
        hist_data = pd.DataFrame({
            'Normal': np.random.normal(0, 1, 1000),
            'Uniform': np.random.uniform(-2, 2, 1000)
        })
        hist_chart = HistogramPyechartsGraph(hist_data, title="测试直方图")
        hist_chart.render(os.path.join(os.path.dirname(__file__), "test_output", "histogram_test.html"))
        print("直方图测试完成")

        # 测试子图
        print("测试子图...")
        subplots_chart = SubplotsPyechartsGraph(
            data[['Series1', 'Series2']],
            title="测试子图",
            rows=1,
            cols=2
        )
        print(os.path.join(os.path.dirname(__file__), "test_output", "subplots_test.html"))
        subplots_chart.render(os.path.join(os.path.dirname(__file__), "test_output", "subplots_test.html"))
        print("子图测试完成")

        print("单个组件测试完成!")

    except ImportError as e:
        print(f"导入pyecharts组件失败: {e}")
    except Exception as e:
        print(f"单个组件测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("开始测试pyecharts_graph组件...")

    # 测试单个组件
    test_individual_components()

    # 测试完整的pyecharts功能
    test_pyecharts_graph()

    # 测试兼容性
    test_compatibility()

    print("\n所有测试完成!")
    print("请查看 test_output 目录中的HTML文件")