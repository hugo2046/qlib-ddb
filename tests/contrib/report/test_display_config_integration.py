"""
报告显示配置集成测试

测试 display_config.py 的所有功能和预定义配置
"""

import sys
sys.path.insert(0, '/data1/hugo/workspace/qlib_ddb')

import numpy as np
import pandas as pd
from pyecharts.commons.utils import JsCode


def test_import_display_config():
    """测试 display_config 模块导入"""
    try:
        from qlib.contrib.report.display_config import (
            LegendConfig,
            TitleConfig,
            GraphDisplayConfig,
            ReportGraphConfig,
            REPORT_DEFAULT_CONFIG,
            IC_GRAPH_CONFIG,
            GROUP_RETURN_CONFIG,
            RISK_ANALYSIS_CONFIG,
            MODEL_PERFORMANCE_CONFIG,
            SCORE_IC_CONFIG
        )
        print("✓ display_config.py 所有配置和类导入成功")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False


def test_predefined_configs():
    """测试所有预定义配置"""
    from qlib.contrib.report.display_config import (
        REPORT_DEFAULT_CONFIG,
        IC_GRAPH_CONFIG,
        GROUP_RETURN_CONFIG,
        RISK_ANALYSIS_CONFIG,
        MODEL_PERFORMANCE_CONFIG,
        SCORE_IC_CONFIG
    )

    configs = {
        "REPORT_DEFAULT_CONFIG": REPORT_DEFAULT_CONFIG,
        "IC_GRAPH_CONFIG": IC_GRAPH_CONFIG,
        "GROUP_RETURN_CONFIG": GROUP_RETURN_CONFIG,
        "RISK_ANALYSIS_CONFIG": RISK_ANALYSIS_CONFIG,
        "MODEL_PERFORMANCE_CONFIG": MODEL_PERFORMANCE_CONFIG,
        "SCORE_IC_CONFIG": SCORE_IC_CONFIG
    }

    all_passed = True
    for name, config in configs.items():
        try:
            # 验证配置结构
            assert hasattr(config, 'legend'), f"{name}: 缺少 legend 属性"
            assert hasattr(config, 'title'), f"{name}: 缺少 title 属性"
            assert hasattr(config, 'to_graph_kwargs'), f"{name}: 缺少 to_graph_kwargs 方法"

            # 验证 to_graph_kwargs() 方法
            kwargs = config.to_graph_kwargs()
            assert isinstance(kwargs, dict), f"{name}: to_graph_kwargs() 应返回 dict"

            print(f"✓ {name}: 配置验证通过")
        except AssertionError as e:
            print(f"✗ {name}: {e}")
            all_passed = False

    return all_passed


def test_score_ic_config():
    """测试 SCORE_IC_CONFIG 的 series_colors 功能"""
    from qlib.contrib.report.display_config import SCORE_IC_CONFIG

    try:
        assert hasattr(SCORE_IC_CONFIG, 'series_colors'), "SCORE_IC_CONFIG 缺少 series_colors 属性"
        assert isinstance(SCORE_IC_CONFIG.series_colors, dict), "series_colors 应该是 dict"
        assert "Rank IC" in SCORE_IC_CONFIG.series_colors, "应该包含 Rank IC 颜色"
        assert SCORE_IC_CONFIG.series_colors["Rank IC"] == "#f0811e", "Rank IC 颜色应该是 #f0811e"

        # 验证 to_graph_kwargs() 包含 series_colors
        kwargs = SCORE_IC_CONFIG.to_graph_kwargs()
        assert "series_colors" in kwargs, "to_graph_kwargs() 应该包含 series_colors"
        assert kwargs["series_colors"]["Rank IC"] == "#f0811e"

        print("✓ SCORE_IC_CONFIG.series_colors: 功能正常")
        return True
    except AssertionError as e:
        print(f"✗ SCORE_IC_CONFIG.series_colors: {e}")
        return False


def test_custom_config():
    """测试自定义配置功能"""
    from qlib.contrib.report.display_config import GraphDisplayConfig, LegendConfig

    try:
        # 创建自定义配置
        custom_config = GraphDisplayConfig(
            legend=LegendConfig(pos_left="80%"),
            height=600
        )

        # 验证配置
        assert custom_config.legend.pos_left == "80%"
        assert custom_config.height == 600

        # 验证转换为 graph_kwargs
        kwargs = custom_config.to_graph_kwargs()
        assert kwargs["legend_pos_left"] == "80%"

        print("✓ 自定义配置: 功能正常")
        return True
    except Exception as e:
        print(f"✗ 自定义配置: {e}")
        return False


def test_report_graph_config():
    """测试 ReportGraphConfig 的特殊配置"""
    from qlib.contrib.report.display_config import ReportGraphConfig

    try:
        config = ReportGraphConfig()

        # 验证图例在右侧
        assert config.legend.pos_left is None, "报告图例应该在右侧（pos_left=None）"
        assert config.legend.pos_right == "5%", "报告图例 pos_right 应该是 5%"

        # 验证包含格式化函数
        assert config.tooltip_formatter is not None, "报告应该有 tooltip_formatter"
        assert config.axis_formatter is not None, "报告应该有 axis_formatter"

        print("✓ ReportGraphConfig: 特殊配置正确")
        return True
    except AssertionError as e:
        print(f"✗ ReportGraphConfig: {e}")
        return False


def test_integration_with_graph():
    """测试与 graph 模块的集成"""
    try:
        from qlib.contrib.report.display_config import SCORE_IC_CONFIG
        from qlib.contrib.report.graph import ScatterGraph

        # 创建测试数据
        data = {
            "IC": [0.1, 0.2, 0.15],
            "Rank IC": [0.08, 0.18, 0.12]
        }
        df = pd.DataFrame(data, index=["2020-01-01", "2020-01-02", "2020-01-03"])

        # 使用配置创建图表
        graph = ScatterGraph(
            df=df,
            layout={"title": "IC Test"},
            graph_kwargs=SCORE_IC_CONFIG.to_graph_kwargs()
        )

        assert graph is not None, "图表创建失败"
        assert hasattr(graph, 'figure'), "图表应该有 figure 属性"

        print("✓ 与 graph 模块集成: 功能正常")
        return True
    except Exception as e:
        print(f"✗ 与 graph 模块集成: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("显示配置集成测试")
    print("=" * 60)
    print()

    results = []

    results.append(("导入测试", test_import_display_config()))
    print()

    results.append(("预定义配置", test_predefined_configs()))
    print()

    results.append(("series_colors 功能", test_score_ic_config()))
    print()

    results.append(("自定义配置", test_custom_config()))
    print()

    results.append(("ReportGraphConfig", test_report_graph_config()))
    print()

    results.append(("graph 模块集成", test_integration_with_graph()))
    print()

    # 汇总结果
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print()
    print(f"通过率: {passed}/{total} ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 所有测试通过！")
        return 0
    else:
        print(f"⚠️  {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
