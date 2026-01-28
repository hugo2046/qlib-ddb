# qlib/contrib/report/display_config.py
"""
报告显示配置模块

集中管理所有报告相关的显示配置,包括:
- 标题位置
- 图例位置
- 颜色配置
- 工具提示格式化
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pyecharts.commons.utils import JsCode
from .graph import (
    get_percent_formatter,
    get_axis_percent_formatter,
    get_number_formatter,
    get_axis_number_formatter,
)


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
    # 报告使用百分比格式化
    tooltip_formatter: Optional[JsCode] = field(
        default_factory=lambda: JsCode(get_percent_formatter(2))
    )
    axis_formatter: Optional[JsCode] = field(
        default_factory=lambda: JsCode(get_axis_percent_formatter(2))
    )


# 预定义配置实例
REPORT_DEFAULT_CONFIG = ReportGraphConfig()

IC_GRAPH_CONFIG = GraphDisplayConfig(
    legend=LegendConfig(pos_left="75%", pos_top="4%"),
    tooltip_formatter=JsCode(get_number_formatter(2)),
)

GROUP_RETURN_CONFIG = GraphDisplayConfig(
    legend=LegendConfig(pos_left="30%", pos_top="4%"),
    tooltip_formatter=JsCode(get_percent_formatter(4)),
)

# 风险分析配置（图例在左侧，高度500）
RISK_ANALYSIS_CONFIG = GraphDisplayConfig(
    legend=LegendConfig(pos_left="0%", pos_top="5%"), height=500
)

# 模型性能配置（图例在右侧70%，使用2位小数格式化，高度400）
MODEL_PERFORMANCE_CONFIG = GraphDisplayConfig(
    legend=LegendConfig(pos_left="70%", pos_top="4%"),
    tooltip_formatter=JsCode(get_number_formatter(2)),
    height=400,
)

# Score IC 配置（图例在右侧75%，Rank IC为橙色，高度400）
SCORE_IC_CONFIG = GraphDisplayConfig(
    legend=LegendConfig(pos_left="75%", pos_top="4%"),
    tooltip_formatter=JsCode(get_number_formatter(2)),
    series_colors={"Rank IC": "#f0811e"},
    height=400,
)

# IC Distribution 配置（隐藏图例，用于 Grid 布局中的子图）
IC_DIST_CONFIG = GraphDisplayConfig(
    legend=LegendConfig(is_show=False),
    tooltip_formatter=JsCode(get_number_formatter(2)),
    height=500,
)

# IC QQ Plot 配置（隐藏图例）
IC_QQ_CONFIG = GraphDisplayConfig(
    legend=LegendConfig(is_show=False),
    tooltip_formatter=JsCode(get_number_formatter(2)),
    height=500,
)


@dataclass
class SubplotsConfig:
    """Subplots 配置"""

    layout: Dict[str, Any] = field(default_factory=dict)
    subplots_kwargs: Dict[str, Any] = field(default_factory=dict)
    kind_map: Dict[str, Any] = field(default_factory=dict)


# Backtest Analysis Report Subplots Config
REPORT_SUBPLOTS_CONFIG = SubplotsConfig(
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

# Risk Analysis Bar Chart Config
RISK_ANALYSIS_SUBPLOTS_CONFIG = SubplotsConfig(
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

# Risk Analysis Monthly Line Chart Config
MONTHLY_RISK_SUBPLOTS_CONFIG = SubplotsConfig(
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

# Model Performance Group Return Config
GROUP_RETURN_SUBPLOTS_CONFIG = SubplotsConfig(
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

# IC Analysis Subplots Config (IC Distribution + QQ Plot)
IC_SUBPLOTS_CONFIG = SubplotsConfig(
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
