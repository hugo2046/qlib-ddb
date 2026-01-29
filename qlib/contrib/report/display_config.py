# qlib/contrib/report/display_config.py
"""
报告显示配置模块

集中管理所有报告相关的显示配置,包括:
- 标题位置
- 图例位置
- 颜色配置
- 工具提示格式化
- 全局图表初始化配置
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pyecharts.commons.utils import JsCode

# 从 graph.py 导入（避免循环引用）
# get_default_init_opts 现在在 graph.py 中定义

@dataclass
class LegendConfig:
    """图例配置"""

    is_show: bool = True  # 是否显示图例
    pos_left: str = "0%"
    pos_right: Optional[str] = None
    pos_top: str = "4%"
    orient: str = "horizontal"


@dataclass
class TitleConfig:
    """标题配置"""

    pos_left: str = "center"
    pos_top: Optional[str] = None


@dataclass
class GraphDisplayConfig:
    """图表显示配置基类"""

    legend: LegendConfig = field(default_factory=LegendConfig)
    title: TitleConfig = field(default_factory=TitleConfig)
    tooltip_formatter: Optional[JsCode] = None
    axis_formatter: Optional[JsCode] = None
    series_colors: Dict[str, str] = field(
        default_factory=dict
    )  # 新增：支持自定义系列颜色
    width: str = "100%"
    height: int = 400

    def to_graph_kwargs(self) -> Dict[str, Any]:
        """转换为 graph_kwargs 字典"""
        result = {}
        if self.legend:
            result.update(
                {
                    "is_show_legend": self.legend.is_show,
                    "legend_pos_left": self.legend.pos_left,
                    "legend_pos_right": self.legend.pos_right,
                    "legend_pos_top": self.legend.pos_top,
                    "legend_orient": self.legend.orient,
                }
            )
        if self.tooltip_formatter:
            result["tooltip_formatter"] = self.tooltip_formatter
        if self.axis_formatter:
            result["axis_formatter"] = self.axis_formatter
        if self.series_colors:
            result["series_colors"] = self.series_colors
        return result


@dataclass
class ReportGraphConfig(GraphDisplayConfig):
    """报告图表专用配置"""

    # 报告的默认图例在右侧
    legend: LegendConfig = field(
        default_factory=lambda: LegendConfig(
            pos_left=None, pos_right="5%", pos_top="4%"
        )
    )
    
    def __post_init__(self):
        """延迟初始化格式化器"""
        if self.tooltip_formatter is None:
            from .graph import get_percent_formatter
            self.tooltip_formatter = JsCode(get_percent_formatter(2))
        if self.axis_formatter is None:
            from .graph import get_axis_percent_formatter
            self.axis_formatter = JsCode(get_axis_percent_formatter(2))


# ==============================================================================
# 预定义配置实例（使用延迟初始化避免循环导入）
# ==============================================================================

# 缓存变量
_cached_configs = {}

def _get_or_create_config(config_name, factory_func):
    """获取或创建配置对象（带缓存）"""
    if config_name not in _cached_configs:
        _cached_configs[config_name] = factory_func()
    return _cached_configs[config_name]

# 报告默认配置
REPORT_DEFAULT_CONFIG = ReportGraphConfig()

# IC 图配置
def _create_ic_graph_config():
    from .graph import get_number_formatter
    return GraphDisplayConfig(
        legend=LegendConfig(pos_left="75%", pos_top="4%"),
        tooltip_formatter=JsCode(get_number_formatter(2)),
    )

IC_GRAPH_CONFIG = _create_ic_graph_config()

# 分组收益配置
def _create_group_return_config():
    from .graph import get_percent_formatter
    return GraphDisplayConfig(
        legend=LegendConfig(pos_left="20%", pos_top="4%"),
        tooltip_formatter=JsCode(get_percent_formatter(4)),
    )

GROUP_RETURN_CONFIG = _create_group_return_config()

# 风险分析配置（图例在左侧，高度500）
RISK_ANALYSIS_CONFIG = GraphDisplayConfig(
    legend=LegendConfig(pos_left="0%", pos_top="5%"), height=500
)

# 模型性能配置
def _create_model_performance_config():
    from .graph import get_number_formatter
    return GraphDisplayConfig(
        legend=LegendConfig(pos_left="70%", pos_top="4%"),
        tooltip_formatter=JsCode(get_number_formatter(2)),
        height=400,
    )

MODEL_PERFORMANCE_CONFIG = _create_model_performance_config()

# Score IC 配置
def _create_score_ic_config():
    from .graph import get_number_formatter
    return GraphDisplayConfig(
        legend=LegendConfig(pos_left="75%", pos_top="4%"),
        tooltip_formatter=JsCode(get_number_formatter(2)),
        series_colors={"Rank IC": "#f0811e"},
        height=400,
    )

SCORE_IC_CONFIG = _create_score_ic_config()

# IC Distribution 配置
def _create_ic_dist_config():
    from .graph import get_number_formatter
    return GraphDisplayConfig(
        legend=LegendConfig(is_show=False),
        tooltip_formatter=JsCode(get_number_formatter(2)),
        height=500,
    )

IC_DIST_CONFIG = _create_ic_dist_config()

# IC QQ Plot 配置
def _create_ic_qq_config():
    from .graph import get_number_formatter
    return GraphDisplayConfig(
        legend=LegendConfig(is_show=False),
        tooltip_formatter=JsCode(get_number_formatter(2)),
        height=500,
    )

IC_QQ_CONFIG = _create_ic_qq_config()

# ============================================================
# 通用可视化函数预设配置
# ============================================================

# 时序图通用配置
def _create_timeseries_config():
    from .graph import get_number_formatter
    return GraphDisplayConfig(
        legend=LegendConfig(pos_right="5%", pos_left=None),
        tooltip_formatter=JsCode(get_number_formatter(4)),
    )

TIMESERIES_CONFIG = _create_timeseries_config()

# 自相关图配置
def _create_autocorr_config():
    from .graph import get_number_formatter
    return GraphDisplayConfig(
        legend=LegendConfig(is_show=False),
        tooltip_formatter=JsCode(get_number_formatter(4)),
    )

AUTOCORR_CONFIG = _create_autocorr_config()

# 换手率图配置
def _create_turnover_config():
    from .graph import get_percent_formatter, get_axis_percent_formatter
    return GraphDisplayConfig(
        legend=LegendConfig(pos_left="75%"),
        tooltip_formatter=JsCode(get_percent_formatter(2)),
        axis_formatter=JsCode(get_axis_percent_formatter(2)),
    )

TURNOVER_CONFIG = _create_turnover_config()

# 分布图配置
def _create_dist_config():
    from .graph import get_number_formatter
    return GraphDisplayConfig(
        legend=LegendConfig(is_show=False),
        tooltip_formatter=JsCode(get_number_formatter(2)),
    )

DIST_CONFIG = _create_dist_config()

# QQ 图配置
def _create_qq_config():
    from .graph import get_number_formatter
    return GraphDisplayConfig(
        legend=LegendConfig(is_show=False),
        tooltip_formatter=JsCode(get_number_formatter(2)),
    )

QQ_CONFIG = _create_qq_config()

# 日历热力图配置
CALENDAR_CONFIG = GraphDisplayConfig(
    legend=LegendConfig(is_show=False),
)


@dataclass
class SubplotsConfig:
    """Subplots 配置"""

    layout: Dict[str, Any] = field(default_factory=dict)
    subplots_kwargs: Dict[str, Any] = field(default_factory=dict)
    kind_map: Dict[str, Any] = field(default_factory=dict)


# Backtest Analysis Report Subplots Config
def _create_report_subplots_config():
    from .graph import get_percent_formatter, get_axis_percent_formatter
    return SubplotsConfig(
        layout=dict(
            height=1200,
            width="100%",
            title="Backtest Analysis Report",
            title_pos_left="center",
        ),
        subplots_kwargs=dict(
            rows=7,
            cols=1,
            row_width=[1, 1, 1, 3, 1, 1, 3],  # Bottom-to-Top
            vertical_spacing=0.03,
            shared_xaxes=True,
        ),
        kind_map=dict(
            kind="ScatterGraph",
            kwargs={
                "mode": "lines+markers",
                "fill": "tozeroy",
                "legend_pos_left": None,
                "legend_pos_right": "5%",
                "tooltip_formatter": JsCode(get_percent_formatter(2)),
                "axis_formatter": JsCode(get_axis_percent_formatter(2)),
                "title_top_offset": -6,
            },
        ),
    )

REPORT_SUBPLOTS_CONFIG = _create_report_subplots_config()

# Risk Analysis Bar Chart Config
def _create_risk_analysis_subplots_config():
    from .graph import get_axis_number_formatter, get_number_formatter
    return SubplotsConfig(
        layout={"height": 460},
        subplots_kwargs={
            "rows": 4,
            "cols": 1,
            "row_width": [1] * 4,
            "vertical_spacing": 0.05,
        },
        kind_map=dict(
            kind="BarGraph",
            kwargs={
                "xy_reverse": False,
                "is_show_label": False,
                "is_show_legend": False,
                "axis_formatter": JsCode(get_axis_number_formatter(3)),
                "tooltip_formatter": JsCode(get_number_formatter(4)),
            },
        ),
    )

RISK_ANALYSIS_SUBPLOTS_CONFIG = _create_risk_analysis_subplots_config()

# Risk Analysis Monthly Line Chart Config
def _create_monthly_risk_subplots_config():
    from .graph import get_percent_formatter, get_axis_percent_formatter
    return SubplotsConfig(
        layout={
            "height": 1000,
            "title": "Monthly Risk Analysis",
            "width": "100%",
            "title_pos_left": "center",
        },
        subplots_kwargs={
            "cols": 1,
            # rows will be set dynamically
            "shared_xaxes": True,
            "vertical_spacing": 0.05,
        },
        kind_map=dict(
            kind="ScatterGraph",
            kwargs={
                "mode": "lines+markers",
                "legend_pos_left": None,
                "legend_pos_right": "5%",
                "legend_pos_top": None,
                "title_top_offset": -4,
                "is_show_legend": True,
                "tooltip_formatter": JsCode(get_percent_formatter(2)),
                "axis_formatter": JsCode(get_axis_percent_formatter(2)),
            },
        ),
    )

MONTHLY_RISK_SUBPLOTS_CONFIG = _create_monthly_risk_subplots_config()

# Model Performance Group Return Config
def _create_group_return_subplots_config():
    from .graph import get_number_formatter, get_axis_number_formatter
    return SubplotsConfig(
        layout=dict(
            height=400,
        ),
        subplots_kwargs=dict(
            rows=1,
            cols=2,
            print_grid=False,
            subplot_titles=["Long-Short", "Long-Average"],
        ),
        kind_map=dict(
            kind="DistplotGraph",
            kwargs=dict(
                is_show_legend=False,
                tooltip_formatter=JsCode(get_number_formatter(decimals=2)),
                axis_formatter=JsCode(get_axis_number_formatter(2)),
                title_top_offset=-6,  # 增加标题向上偏移量，避免与绘图区重叠
                axis_pointer_type="shadow",
            ),
        ),  # kwargs will be updated with bin_size
    )

GROUP_RETURN_SUBPLOTS_CONFIG = _create_group_return_subplots_config()

# IC Analysis Subplots Config (IC Distribution + QQ Plot)
def _create_ic_subplots_config():
    from .graph import get_number_formatter, get_axis_number_formatter
    return SubplotsConfig(
        layout=dict(
            height=500,
            width="100%",
        ),
        subplots_kwargs=dict(
            rows=1,
            cols=2,
            print_grid=False,
            subplot_titles=["IC Distribution", "IC Normal Dist. Q-Q"],
        ),
        kind_map=dict(
            kind="DistplotGraph",  # Default kind
            kwargs=dict(
                is_show_legend=False,
                tooltip_formatter=JsCode(get_number_formatter(decimals=2)),
                axis_formatter=JsCode(get_axis_number_formatter(2)),
                title_top_offset=-6,
                axis_pointer_type="shadow",
            ),
        ),
    )

IC_SUBPLOTS_CONFIG = _create_ic_subplots_config()


# Analysis Model Performance Layouts
IC_HEATMAP_LAYOUT = {
    "title": "Monthly IC",
    "xaxis": {"name": "Month"},
    "yaxis": {"name": "Year"},
}

IC_DIST_LAYOUT = {
    "title": "IC Distribution",
    "title_pos_left": "20%",
    "width": "100%",
    "height": 500,
}

IC_QQ_LAYOUT = {
    "title": "IC Normal Dist. Q-Q",
    "title_pos_left": "65%",
    "width": "100%",
    "height": 500,
    "xaxis": {"title": "Normal Distribution Quantile"},
    "yaxis": {"title": "Observed Quantile"},
}

IC_CALENDAR_LAYOUT = {
    "title": "Daily IC",
    "width": "100%",
    "height": "600px",
}
