"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2026-01-16 16:57:25
LastEditors: shen.lan123@gmail.com
LastEditTime: 2026-01-25 21:00:00
Description: pyecharts重构画图
"""

import importlib
import math
import os
from typing import Any, Iterable, List, Optional

import numpy as np
import pandas as pd
from pyecharts import options as opts

# 引入 Pyecharts 组件
from pyecharts.charts import Bar, Grid, HeatMap, Line
from pyecharts.commons.utils import JsCode  # [New] 支持 JS Formatter
from pyecharts.globals import ThemeType
from scipy.stats import gaussian_kde

# ============================================================================
# Jupyter 环境检测与适配
# ============================================================================


class _JupyterEnvironmentDetector:
    """Jupyter 环境检测器 (内部类)

    用于检测当前运行环境是 Jupyter Notebook 还是 Jupyter Lab,
    以便正确配置 pyecharts 的显示参数。

    参考:
    - https://github.com/pyecharts/pyecharts/issues/1756
    - https://blog.csdn.net/silent1cat/article/details/117944987
    """

    _detected_type = None
    _is_loaded = False
    _env_type = None

    @classmethod
    def _detect_from_process_tree(cls) -> Optional[str]:
        """从进程树中检测 Jupyter 环境

        使用 psutil 向上遍历进程树，查找 jupyter-lab 或 jupyter-notebook 进程。
        这是最可靠的检测方法，因为它直接检查父进程。

        Returns:
            "jupyter_lab", "jupyter_notebook", 或 None
        """
        try:
            import psutil
        except ImportError:
            raise ImportError("自动检测需要 psutil 库，请先安装：pip install psutil")

        try:
            cls._env_type = psutil.Process().parent().name()
            return cls._env_type

        except (ImportError, Exception):
            # psutil 不可用或检测失败
            raise ValueError("通过检查进程树检测 Jupyter 环境时出错。")

    @classmethod
    def configure_pyecharts(cls):
        """配置 pyecharts 以适配当前的 Jupyter 环境

        对于 Jupyter Lab 环境,需要设置 CurrentConfig.NOTEBOOK_TYPE。
        该方法会自动检测环境并进行配置。
        """
        if cls._is_loaded:
            return
        env_type = cls._detect_from_process_tree()

        if env_type == "jupyter-lab":
            try:
                from pyecharts.globals import CurrentConfig, NotebookType

                CurrentConfig.ONLINE_HOST = "https://assets.pyecharts.org/assets/"
                CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_LAB
            except ImportError:
                pass  # pyecharts 未安装或版本不支持

    @classmethod
    def load_javascript_if_needed(cls, chart):
        """在需要时加载 pyecharts JavaScript

        对于 Jupyter Lab,首次渲染前建议调用 load_javascript()。
        该方法会确保只加载一次。
        """
        if cls._env_type == "jupyter-lab":
            chart.load_javascript()


# ============================================================================
# Pyecharts 工具函数区
# ============================================================================


def get_percent_formatter(decimals: int = 4) -> str:
    """生成 Tooltip 百分比格式化 JavaScript 代码

    该函数生成一段 JavaScript 代码，用于 ECharts Tooltip 的 formatter 配置。
    支持自动处理 trigger="axis" (params 为数组) 和 trigger="item" (params 为对象) 两种模式。
    数据会在前端自动乘以 100 并保留指定小数位，然后添加百分号。

    .. note::
        此函数返回的是 JavaScript 函数字符串，需要配合 :class:`pyecharts.commons.utils.JsCode` 使用。
        数据应保持原始小数形式（如 0.2345），由前端 JS 负责转换为百分比显示。

    Args:
        decimals (int): 小数位数，默认为 4 位。例如：
            - decimals=0: 显示为 "23 %"
            - decimals=2: 显示为 "23.45 %"
            - decimals=4: 显示为 "23.4500 %"

    Returns:
        str: JavaScript 函数字符串，可直接传递给 :class:`~pyecharts.options.TooltipOpts` 的 formatter 参数。

    Raises:
        无异常

    Example:
        >>> from pyecharts.commons.utils import JsCode
        >>> from pyecharts import options as opts
        >>>
        >>> # 方式 1: 直接使用字符串（BaseGraph 会自动检测并转换为 JsCode）
        >>> tooltip_formatter = get_percent_formatter(decimals=2)
        >>>
        >>> # 方式 2: 显式包装为 JsCode（推荐）
        >>> tooltip_opts = opts.TooltipOpts(
        ...     trigger="axis",
        ...     formatter=JsCode(get_percent_formatter(decimals=2))
        ... )
        >>>
        >>> # 在 graph_kwargs 中使用
        >>> graph_kwargs = {
        ...     "tooltip_formatter": JsCode(get_percent_formatter(4))
        ... }

    See Also:
        :func:`get_axis_percent_formatter`: 生成坐标轴百分比格式化器
        :meth:`BaseGraph._normalize_formatter`: 智能检测并转换 formatter
    """
    # [Fix] 使用单行格式避免HTML生成时的换行和编码问题
    return f"function(params){{var res=params[0].name+'<br/>';for(var i=0;i<params.length;i++){{var val=params[i].value;if(Array.isArray(val)){{val=val[1];}}var pct=(Number(val)*100).toFixed({decimals})+'%';res+=params[i].marker+params[i].seriesName+':'+pct+'<br/>';}}return res;}}"


def get_axis_percent_formatter(decimals: int = 0) -> str:
    """生成坐标轴百分比格式化 JavaScript 代码

    该函数生成一段 JavaScript 代码，用于 ECharts 坐标轴标签的 formatter 配置。
    数据会在前端自动乘以 100 并保留指定小数位，然后添加百分号。

    .. note::
        此函数返回的是 JavaScript 函数字符串，需要配合 :class:`pyecharts.commons.utils.JsCode` 使用。
        适用于 Y 轴或 X 轴标签的百分比显示。

    Args:
        decimals (int): 小数位数，默认为 0 位（整数显示）。例如：
            - decimals=0: 显示为 "23 %"（默认，整数）
            - decimals=1: 显示为 "23.5 %"
            - decimals=2: 显示为 "23.45 %"

    Returns:
        str: JavaScript 函数字符串，可直接传递给 :class:`~pyecharts.options.LabelOpts` 的 formatter 参数。

    Raises:
        无异常

    Example:
        >>> from pyecharts.commons.utils import JsCode
        >>> from pyecharts import options as opts
        >>>
        >>> # 方式 1: 在 graph_kwargs 中使用
        >>> graph_kwargs = {
        ...     "axis_formatter": JsCode(get_axis_percent_formatter(decimals=2))
        ... }
        >>>
        >>> # 方式 2: 直接配置到 AxisOpts
        >>> yaxis_opts = opts.AxisOpts(
        ...     is_scale=True,
        ...     axislabel_opts=opts.LabelOpts(
        ...         formatter=JsCode(get_axis_percent_formatter(0))
        ...     )
        ... )

    See Also:
        :func:`get_percent_formatter`: 生成 Tooltip 百分比格式化器
        :meth:`BaseGraph._normalize_formatter`: 智能检测并转换 formatter
    """
    return f"function(value) {{ return (value * 100).toFixed({decimals}) + ' %'; }}"


def get_number_formatter(decimals: int = 2, use_comma: bool = True) -> str:
    """生成 Tooltip 数字格式化 JavaScript 代码

    该函数生成一段 JavaScript 代码，用于 ECharts Tooltip 的 formatter 配置。
    支持自动处理 trigger="axis" (params 为数组) 和 trigger="item" (params 为对象) 两种模式。
    数值会保留指定小数位，并可选择是否使用千分位分隔符。

    .. note::
        此函数返回的是 JavaScript 函数字符串，需要配合 :class:`pyecharts.commons.utils.JsCode` 使用。
        适用于普通数值（非百分比）的格式化显示。

    Args:
        decimals (int): 小数位数，默认为 2 位。例如：
            - decimals=0: 显示为 "1,234"
            - decimals=2: 显示为 "1,234.57"
            - decimals=4: 显示为 "1,234.5678"
        use_comma (bool): 是否使用千分位分隔符，默认为 True。
            - True: "1,234.57"
            - False: "1234.57"

    Returns:
        str: JavaScript 函数字符串，可直接传递给 :class:`~pyecharts.options.TooltipOpts` 的 formatter 参数。

    Raises:
        无异常

    Example:
        >>> from pyecharts.commons.utils import JsCode
        >>> from pyecharts import options as opts
        >>>
        >>> # 方式 1: 直接使用字符串（BaseGraph 会自动检测并转换为 JsCode）
        >>> tooltip_formatter = get_number_formatter(decimals=2, use_comma=True)
        >>>
        >>> # 方式 2: 显式包装为 JsCode（推荐）
        >>> tooltip_opts = opts.TooltipOpts(
        ...     trigger="axis",
        ...     formatter=JsCode(get_number_formatter(decimals=2))
        ... )
        >>>
        >>> # 在 graph_kwargs 中使用
        >>> graph_kwargs = {
        ...     "tooltip_formatter": JsCode(get_number_formatter(decimals=4))
        ... }

    See Also:
        :func:`get_percent_formatter`: 生成百分比格式化器
        :func:`get_axis_percent_formatter`: 生成坐标轴百分比格式化器
        :func:`get_axis_number_formatter`: 生成坐标轴数字格式化器
    """
    if use_comma:
        # 使用千分位分隔符
        return f"function(params){{var res=params[0].name+'<br/>';for(var i=0;i<params.length;i++){{var val=params[i].value;if(Array.isArray(val)){{val=val[1];}}var num=Number(val).toFixed({decimals});var parts=num.split('.');parts[0]=parts[0].replace(/\\B(?=(\\d{{3}})+(?!\\d))/g,',');var formatted=parts.join('.');res+=params[i].marker+params[i].seriesName+': '+formatted+'<br/>';}}return res;}}"
    else:
        # 不使用千分位分隔符
        return f"function(params){{var res=params[0].name+'<br/>';for(var i=0;i<params.length;i++){{var val=params[i].value;if(Array.isArray(val)){{val=val[1];}}var num=Number(val).toFixed({decimals});res+=params[i].marker+params[i].seriesName+': '+num+'<br/>';}}return res;}}"


def get_axis_number_formatter(decimals: int = 2, use_comma: bool = True) -> str:
    """生成坐标轴数字格式化 JavaScript 代码

    该函数生成一段 JavaScript 代码，用于 ECharts 坐标轴标签的 formatter 配置。
    数值会保留指定小数位，并可选择是否使用千分位分隔符。

    .. note::
        此函数返回的是 JavaScript 函数字符串，需要配合 :class:`pyecharts.commons.utils.JsCode` 使用。
        适用于 Y 轴或 X 轴标签的数字显示。

    Args:
        decimals (int): 小数位数，默认为 2 位。例如：
            - decimals=0: 显示为 "1,234"（整数）
            - decimals=1: 显示为 "1,234.6"
            - decimals=2: 显示为 "1,234.57"
        use_comma (bool): 是否使用千分位分隔符，默认为 True。
            - True: "1,234.57"
            - False: "1234.57"

    Returns:
        str: JavaScript 函数字符串，可直接传递给 :class:`~pyecharts.options.LabelOpts` 的 formatter 参数。

    Raises:
        无异常

    Example:
        >>> from pyecharts.commons.utils import JsCode
        >>> from pyecharts import options as opts
        >>>
        >>> # 方式 1: 在 graph_kwargs 中使用
        >>> graph_kwargs = {
        ...     "axis_formatter": JsCode(get_axis_number_formatter(decimals=0))
        ... }
        >>>
        >>> # 方式 2: 直接配置到 AxisOpts
        >>> yaxis_opts = opts.AxisOpts(
        ...     is_scale=True,
        ...     axislabel_opts=opts.LabelOpts(
        ...         formatter=JsCode(get_axis_number_formatter(decimals=2))
        ...     )
        ... )

    See Also:
        :func:`get_number_formatter`: 生成 Tooltip 数字格式化器
        :func:`get_axis_percent_formatter`: 生成坐标轴百分比格式化器
        :meth:`BaseGraph._normalize_formatter`: 智能检测并转换 formatter
    """
    if use_comma:
        # 使用千分位分隔符
        return f"function(value) {{ var num = Number(value).toFixed({decimals}); var parts = num.split('.'); parts[0] = parts[0].replace(/\\B(?=(\\d{{3}})+(?!\\d))/g, ','); return parts.join('.'); }}"
    else:
        # 不使用千分位分隔符
        return f"function(value) {{ return Number(value).toFixed({decimals}); }}"


class BaseGraph:
    _name = None

    def __init__(
        self,
        df: pd.DataFrame = None,
        layout: dict = None,
        graph_kwargs: dict = None,
        name_dict: dict = None,
        config: Any = None,  # [Refactor] Add config parameter
        **kwargs,
    ):
        self._df = df

        # [Refactor] Process config
        config_layout = {}
        config_graph_kwargs = {}

        if config:
            # 1. Extract layout params
            if hasattr(config, "width") and config.width:
                config_layout["width"] = config.width
            if hasattr(config, "height") and config.height:
                config_layout["height"] = config.height
            if hasattr(config, "title"):
                # title config mapping might be complex, assuming standard struct
                pass
                # BaseGraph usually expects 'title' key in layout for simple string,
                # or specific title handling properties?
                # Actually BaseGraph doesn't seem to use 'title' in layout directly in _init_chart?
                # It depends on subclass implementation.
                # But let's check config.to_graph_kwargs() logic

            # 2. Extract graph_kwargs
            if hasattr(config, "to_graph_kwargs"):
                config_graph_kwargs = config.to_graph_kwargs()

        # Merge layout: kwargs > config
        self._layout = config_layout
        if layout:
            self._layout.update(layout)

        # Merge graph_kwargs: kwargs > config
        self._graph_kwargs = config_graph_kwargs
        if graph_kwargs:
            self._graph_kwargs.update(graph_kwargs)

        self._name_dict = name_dict

        self.chart = None

        self._init_parameters(**kwargs)
        self._init_data()

    def _init_data(self):
        if self._df is None or self._df.empty:
            pass
        self._init_chart()

    def _init_parameters(self, **kwargs):
        self._graph_type = self._name.lower().capitalize()
        if self._name_dict is None and self._df is not None:
            self._name_dict = {_item: _item for _item in self._df.columns}

    def _init_chart(self):
        raise NotImplementedError

    @staticmethod
    def get_instance_with_graph_parameters(graph_type: str = None, **kwargs):
        try:
            if not graph_type.endswith("Graph"):
                graph_class_name = f"{graph_type}Graph"
            else:
                graph_class_name = graph_type

            if graph_class_name in globals():
                _graph_class = globals()[graph_class_name]
            else:
                _graph_module = importlib.import_module("qlib.contrib.report.graph")
                _graph_class = getattr(_graph_module, graph_class_name)

        except (AttributeError, ImportError):
            _graph_class = ScatterGraph

        return _graph_class(**kwargs)

    @staticmethod
    def show_graph_in_notebook(figure_list: Iterable = None):
        """在 Jupyter Notebook 或 Jupyter Lab 中显示 pyecharts 图表

        该方法会自动检测 Jupyter 环境类型 (Notebook / Lab) 并进行相应配置:
        - Jupyter Notebook: 直接调用 render_notebook()
        - Jupyter Lab: 自动配置 CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_LAB

        Args:
            figure_list (Iterable): pyecharts 图表对象的可迭代对象

        使用示例:
            >>> from qlib.contrib.report.graph import show_graph_in_notebook
            >>> from pyecharts.charts import Line
            >>> line = Line()
            >>> # ... 配置图表 ...
            >>> show_graph_in_notebook([line])

        参考:
            - https://github.com/pyecharts/pyecharts/issues/1756
            - https://blog.csdn.net/silent1cat/article/details/117944987
        """
        if figure_list is None:
            return

        # 配置 pyecharts 以适配当前的 Jupyter 环境
        _JupyterEnvironmentDetector.configure_pyecharts()

        # 尝试加载 JavaScript (对于 Jupyter Lab)
        for _chart in figure_list:
            if hasattr(_chart, "render_notebook"):
                try:
                    from IPython.display import display as ipy_display

                    _JupyterEnvironmentDetector.load_javascript_if_needed(_chart)
                    rendered = _chart.render_notebook()
                    ipy_display(rendered)
                except ImportError:
                    print("IPython not found, cannot render in notebook.")
            else:
                print(
                    f"Warning: Object does not have render_notebook() method: {type(_chart)}"
                )

    @staticmethod
    def _normalize_formatter(formatter):
        """标准化 formatter，支持字符串模板和 JavaScript 函数字符串

        该方法用于智能检测并转换 formatter 参数，实现向后兼容性。
        支持三种输入格式：
        1. None: 直接返回 None
        2. 字符串模板（如 "{value} %"）: 直接返回，用于 ECharts 原生字符串模板
        3. JavaScript 函数字符串: 自动包装为 :class:`~pyecharts.commons.utils.JsCode` 对象

        .. note::
            此方法通过检测字符串中是否包含 "function" 关键字来判断是否为 JavaScript 函数。
            这种设计允许用户既可以使用简单的字符串模板，也可以使用复杂的 JavaScript 格式化逻辑。

        Args:
            formatter: 可以是以下类型之一：
                - None: 不使用格式化器
                - str: 字符串模板（如 "{value} %"）或 JavaScript 函数字符串（如 "function(x){return x;}"）
                - JsCode: 已包装的 JsCode 对象（直接返回）

        Returns:
            None or str or JsCode: 根据输入类型返回：
                - None: 输入为 None 时
                - str: 输入为字符串模板时
                - JsCode: 输入为 JavaScript 函数字符串时（自动包装）

        Raises:
            无异常

        Example:
            >>> from pyecharts.commons.utils import JsCode
            >>>
            >>> # 案例 1: 字符串模板（用于简单的占位符替换）
            >>> _normalize_formatter("{value} %")
            '{value} %'
            >>>
            >>> # 案例 2: JavaScript 函数字符串（自动包装为 JsCode）
            >>> result = _normalize_formatter("function(x){return x * 100;}")
            >>> isinstance(result, JsCode)
            True
            >>>
            >>> # 案例 3: None 直接返回
            >>> _normalize_formatter(None) is None
            True
            >>>
            >>> # 案例 4: JsCode 对象直接返回
            >>> js_code = JsCode("function(x){return x;}")
            >>> _normalize_formatter(js_code) is js_code
            True

        See Also:
            :func:`get_percent_formatter`: 生成 Tooltip 百分比格式化器
            :func:`get_axis_percent_formatter`: 生成坐标轴百分比格式化器
        """
        if formatter is None:
            return None
        if isinstance(formatter, str) and "function" in formatter:
            return JsCode(formatter)
        return formatter

    def _apply_global_opts(self):
        if not self.chart:
            return

        # 1. 获取基础配置
        is_show_legend = self._graph_kwargs.get("is_show_legend", True)
        legend_pos_top = self._graph_kwargs.get("legend_pos_top", "5%")
        legend_pos_left = self._graph_kwargs.get("legend_pos_left", "0%")
        legend_pos_right = self._graph_kwargs.get("legend_pos_right", None)
        legend_orient = self._graph_kwargs.get("legend_orient", "horizontal")

        if legend_pos_right is not None:
            legend_pos_left = None

        # [New] 获取 legend_data
        legend_data = self._graph_kwargs.get("legend_data", None)

        # [New] 标准化 formatter（支持字符串和 JsCode）
        axis_formatter = self._normalize_formatter(
            self._graph_kwargs.get("axis_formatter", None)
        )
        tooltip_formatter = self._normalize_formatter(
            self._graph_kwargs.get("tooltip_formatter", None)
        )

        yaxis_opts = opts.AxisOpts(
            is_scale=True,
            axislabel_opts=opts.LabelOpts(formatter=axis_formatter),
            splitline_opts=opts.SplitLineOpts(
                is_show=True,
                linestyle_opts=opts.LineStyleOpts(opacity=0.5, type_="dashed"),
            ),
        )

        # 2. 应用配置 (支持 title_pos_left 和 orient)

        # [Fix] LegendOpts doesn't support 'data' in __init__. We inject it manually.
        _legend_opts = opts.LegendOpts(
            is_show=is_show_legend,
            pos_top=legend_pos_top,
            pos_left=legend_pos_left,
            pos_right=legend_pos_right,
            orient=legend_orient,  # <--- 支持图例方向配置
        )
        if legend_data:
            _legend_opts.opts["data"] = legend_data

        self.chart.set_global_opts(
            title_opts=opts.TitleOpts(
                title=self._layout.get("title", ""),
                pos_left=self._layout.get("title_pos_left", "center"),  # <--- 默认居中
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
                axis_pointer_type="line",
                formatter=tooltip_formatter,  # [New] 应用 formatter
            ),
            legend_opts=_legend_opts,
            xaxis_opts=opts.AxisOpts(
                type_="category",
                is_scale=True,
                splitline_opts=opts.SplitLineOpts(
                    is_show=True,
                    linestyle_opts=opts.LineStyleOpts(opacity=0.5, type_="dashed"),
                ),
            ),
            yaxis_opts=yaxis_opts,
        )

    @property
    def figure(self):
        self._apply_global_opts()
        return self.chart


class ScatterGraph(BaseGraph):
    _name = "scatter"

    def _init_chart(self):
        self.chart = Line()
        x_data = self._df.index.astype(str).tolist()
        self.chart.add_xaxis(x_data)

        for col, name in self._name_dict.items():
            raw_data = self._df[col].tolist()
            # [Pure] 仅负责处理 NaN
            y_data = [x if pd.notna(x) else None for x in raw_data]

            final_name = name

            areastyle_opts = None
            if self._graph_kwargs.get("fill") == "tozeroy":
                areastyle_opts = opts.AreaStyleOpts(opacity=0.3)

            markarea_opts = self._graph_kwargs.get("markarea_opts", None)
            mode = self._graph_kwargs.get("mode", "lines")
            is_symbol_show = "markers" in mode

            # 支持自定义颜色（通过 series_colors 字典传递）
            series_colors = self._graph_kwargs.get("series_colors", {})
            custom_color = series_colors.get(final_name, None)

            # 构建基础参数
            add_yaxis_kwargs = {
                "series_name": final_name,
                "y_axis": y_data,
                "is_symbol_show": is_symbol_show,
                "areastyle_opts": areastyle_opts,
                "markarea_opts": markarea_opts,
                "label_opts": opts.LabelOpts(is_show=False),
                "is_smooth": False,
            }

            # 如果指定了颜色，设置 linestyle_opts（Line 图表使用 linestyle_opts）
            # 同时设置 itemstyle_opts 以确保图例颜色正确显示
            if custom_color is not None:
                add_yaxis_kwargs["linestyle_opts"] = opts.LineStyleOpts(
                    color=custom_color
                )
                # itemstyle_opts 确保图例、标记点等元素使用相同颜色
                add_yaxis_kwargs["itemstyle_opts"] = opts.ItemStyleOpts(
                    color=custom_color
                )

            self.chart.add_yaxis(**add_yaxis_kwargs)


class BarGraph(BaseGraph):
    _name = "bar"

    def _init_chart(self):
        self.chart = Bar()
        x_data = self._df.index.astype(str).tolist()

        self.chart.add_xaxis(x_data)

        if self._graph_kwargs.get("xy_reverse", False):
            self.chart.reversal_axis()

        for col, name in self._name_dict.items():
            raw_data = self._df[col].tolist()
            # [Pure] 仅负责处理 NaN
            y_data = [x if pd.notna(x) else None for x in raw_data]

            final_name = name

            # [Feature] Support custom color for BarGraph
            itemstyle_opts = None
            if self._graph_kwargs.get("color"):
                itemstyle_opts = opts.ItemStyleOpts(
                    color=self._graph_kwargs.get("color")
                )

            self.chart.add_yaxis(
                series_name=final_name,
                y_axis=y_data,
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=itemstyle_opts,
            )


class DistplotGraph(BaseGraph):
    """
    分布图：直方图 + 核密度估计曲线 (模拟 Plotly 的 distplot)
    """

    _name = "distplot"

    def _init_chart(self):
        # 初始化一个 Bar 作为基础，因为 Bar 拥有 X 轴
        self.chart = Bar()

        # 获取配置参数
        bin_size = self._graph_kwargs.get("bin_size", None)

        # 1. 计算所有数据的范围，确保 X 轴对齐 (统一 Bins)
        _min_val, _max_val = float("inf"), float("-inf")
        valid_dfs = {}

        for col, name in self._name_dict.items():
            data = self._df[col].dropna().values
            if len(data) == 0:
                continue
            _min_val = min(_min_val, data.min())
            _max_val = max(_max_val, data.max())
            valid_dfs[name] = data

        if not valid_dfs:
            return

        # 确定 bins
        if bin_size:
            bins = np.arange(np.floor(_min_val), np.ceil(_max_val) + bin_size, bin_size)
        else:
            bins = 50  # 默认分50份

        # 2. 生成 X 轴坐标 (使用 numpy histogram 的 bin edges)
        combined_data = np.concatenate(list(valid_dfs.values()))
        hist_total, bin_edges = np.histogram(combined_data, bins=bins)

        # X 轴显示为 bin 的中心点
        x_axis_str = [
            f"{(bin_edges[i] + bin_edges[i+1])/2:.4f}"
            for i in range(len(bin_edges) - 1)
        ]

        self.chart.add_xaxis(x_axis_str)

        # 3. 循环添加系列
        for name, data in valid_dfs.items():
            # A. 直方图数据
            # density=True 让直方图的高度归一化，以便和 KDE 曲线匹配
            hist, _ = np.histogram(data, bins=bin_edges, density=True)

            # 添加直方图 (Bar)
            self.chart.add_yaxis(
                series_name=f"{name} (Hist)",
                y_axis=hist.tolist(),
                category_gap=0,  # 让柱子紧挨着
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(opacity=0.3),  # 半透明
                z=0,  # 图层靠后
            )

            # B. 核密度估计 (KDE Line)
            try:
                kde = gaussian_kde(data)
                # 在 X 轴对应的点上计算 PDF 值
                x_points = (bin_edges[:-1] + bin_edges[1:]) / 2
                y_kde = kde(x_points)

                # 叠加 Line 图
                line = (
                    Line()
                    .add_xaxis(x_axis_str)
                    .add_yaxis(
                        series_name=f"{name} (KDE)",
                        y_axis=y_kde.tolist(),
                        is_smooth=True,  # 平滑曲线
                        symbol="none",  # 不显示点
                        label_opts=opts.LabelOpts(is_show=False),
                        z=10,  # 图层靠前
                    )
                )
                self.chart.overlap(line)
            except Exception:
                pass

    def _apply_global_opts(self):
        """Distplot 特有的全局配置"""
        if not self.chart:
            return
        super()._apply_global_opts()

        # 获取用户自定义的 tooltip_formatter（如果有）
        tooltip_formatter = self._normalize_formatter(
            self._graph_kwargs.get("tooltip_formatter", None)
        )

        # 强制更新一些适合 Distplot 的配置
        self.chart.set_global_opts(
            xaxis_opts=opts.AxisOpts(
                type_="category",
                is_scale=True,
                axislabel_opts=opts.LabelOpts(rotate=45),
                name_location="middle",
                name_gap=30,
                splitline_opts=opts.SplitLineOpts(
                    is_show=True,
                    linestyle_opts=opts.LineStyleOpts(opacity=0.5, type_="dashed"),
                ),
            ),
            # 如果用户提供了自定义 formatter，使用用户的；否则使用默认的 shadow 模式
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
                axis_pointer_type="shadow",
                formatter=tooltip_formatter,
            ),
        )


class HeatmapGraph(BaseGraph):
    _name = "heatmap"

    def _init_chart(self):
        self.chart = HeatMap()
        x_axis = self._df.columns.tolist()
        y_axis = self._df.index.astype(str).tolist()
        self.chart.add_xaxis(x_axis)
        self.chart.add_yaxis(
            series_name="",
            yaxis_data=y_axis,
            value=[
                [i, j, float(self._df.iloc[j, i])]
                for i in range(len(x_axis))
                for j in range(len(y_axis))
            ],
            label_opts=opts.LabelOpts(is_show=True, position="inside"),
        )
        self.chart.set_global_opts(visualmap_opts=opts.VisualMapOpts(is_show=True))


class HistogramGraph(BaseGraph):
    _name = "histogram"

    def _init_chart(self):
        self.chart = Bar()
        x_data = self._df.index.astype(str).tolist()
        self.chart.add_xaxis(x_data)
        for col, name in self._name_dict.items():
            y_data = [
                x * 100 if not pd.isna(x) else None for x in self._df[col].tolist()
            ]
            self.chart.add_yaxis(
                series_name=f"{name} (%)",
                y_axis=y_data,
                category_gap=0,
                label_opts=opts.LabelOpts(is_show=False),
            )


class SubplotsGraph:
    """子图管理器，使用 Pyecharts Grid 实现多子图布局

    该类提供类似于 ``df.plot(subplots=True)`` 的功能，支持在一个画布上创建多个子图。
    每个子图可以是不同类型的图表（如 ScatterGraph、BarGraph、DistplotGraph 等），
    并且支持灵活的行列布局、共享坐标轴等高级功能。

    .. note::
        子图的图表类型通过 ``kind_map`` 或 ``config`` 参数指定，支持动态实例化不同的 Graph 类。
        布局参数（如标题位置、图例位置）会自动计算以适配 Grid 布局。

    Args:
        df (pd.DataFrame, optional): 数据源 DataFrame，每列对应一个子图的数据系列。默认为 None。
        kind_map (dict, optional): 图表类型映射配置，包含 ``kind`` (图表类型名称) 和 ``kwargs`` (图表参数)。
            例如: ``{"kind": "ScatterGraph", "kwargs": {"mode": "lines"}}``。默认为 None。
        layout (dict, optional): 全局布局配置，如 ``{"height": 1000, "width": "100%", "title": "总标题"}``。
            默认为 None。
        sub_graph_layout (dict, optional): 子图布局配置（保留参数，当前未使用）。默认为 None。
        sub_graph_data (list, optional): 手动指定的子图数据配置列表。每个元素为 ``(列名, 配置字典)`` 元组，
            配置字典包含 ``row``, ``col``, ``name``, ``title``, ``kind``, ``graph_kwargs`` 等字段。
            如果未提供，将根据 ``df`` 自动生成。默认为 None。
        subplots_kwargs (dict, optional): 子图布局参数，包含:

            - ``rows`` (int): 行数
            - ``cols`` (int): 列数
            - ``shared_xaxes`` (bool): 是否共享 X 轴
            - ``vertical_spacing`` (float): 垂直间距比例（0-1）
            - ``horizontal_spacing`` (float): 水平间距比例（0-1）
            - ``row_width`` (list): 各行高度权重列表
            - ``col_width`` (list): 各列宽度权重列表

            默认为 None。
        config (Any, optional): 配置对象（如 ``SubplotsConfig``），包含 ``layout``, ``subplots_kwargs``,
            ``kind_map`` 属性。如果提供，将作为默认配置，可被其他参数覆盖。默认为 None。
        **kwargs: 其他关键字参数（保留用于扩展）。

    Attributes:
        _df (pd.DataFrame): 数据源 DataFrame
        _layout (dict): 合并后的全局布局配置
        _kind_map (dict): 合并后的图表类型映射配置
        _subplots_kwargs (dict): 合并后的子图布局参数
        _sub_graph_data (list): 子图数据配置列表
        _grid (Grid): Pyecharts Grid 对象

    Examples:
        >>> import pandas as pd
        >>> from qlib.contrib.report.graph import SubplotsGraph
        >>> from qlib.contrib.report.display_config import REPORT_SUBPLOTS_CONFIG
        >>>
        >>> # 示例 1: 使用预定义配置
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        >>> sg = SubplotsGraph(df=df, config=REPORT_SUBPLOTS_CONFIG)
        >>> sg.figure.render("output.html")
        >>>
        >>> # 示例 2: 手动指定子图配置
        >>> sub_graph_data = [
        ...     ("A", {"row": 1, "col": 1, "title": "Series A"}),
        ...     ("B", {"row": 1, "col": 2, "title": "Series B"}),
        ... ]
        >>> sg = SubplotsGraph(
        ...     df=df,
        ...     sub_graph_data=sub_graph_data,
        ...     subplots_kwargs={"rows": 1, "cols": 2},
        ...     kind_map={"kind": "ScatterGraph", "kwargs": {"mode": "lines"}},
        ... )

    See Also:
        :class:`BaseGraph`: 图表基类
        :class:`ScatterGraph`: 折线图/散点图
        :class:`BarGraph`: 柱状图
        :class:`DistplotGraph`: 分布图（直方图 + KDE）
    """

    def __init__(
        self,
        df: pd.DataFrame = None,
        kind_map: dict = None,
        layout: dict = None,
        sub_graph_layout: dict = None,
        sub_graph_data: list = None,
        subplots_kwargs: dict = None,
        config: Any = None,
        **kwargs,
    ):
        self._df = df

        # [Refactor] Unpack config if provided
        config_layout = {}
        config_subplots_kwargs = {}
        config_kind_map = None

        if config:
            if hasattr(config, "layout"):
                config_layout = config.layout
            if hasattr(config, "subplots_kwargs"):
                config_subplots_kwargs = config.subplots_kwargs
            if hasattr(config, "kind_map"):
                config_kind_map = config.kind_map

        # kwargs takes precedence over config (Merge recursively or shallowly)
        # Here we perform shallow merge: existing keys in kwargs overwrite config, new keys are added.
        self._layout = config_layout.copy()
        if layout:
            self._layout.update(layout)

        self._sub_graph_layout = sub_graph_layout or {}

        # Merge kind_map
        # Determine base kind_map
        base_kind_map = (
            config_kind_map
            if config_kind_map is not None
            else dict(kind="ScatterGraph", kwargs=dict())
        )

        if kind_map:
            self._kind_map = base_kind_map.copy()
            if "kind" in kind_map:
                self._kind_map["kind"] = kind_map["kind"]
            if "kwargs" in kind_map:
                # Update kwargs
                if "kwargs" not in self._kind_map:
                    self._kind_map["kwargs"] = {}
                self._kind_map["kwargs"].update(kind_map["kwargs"])
        else:
            self._kind_map = base_kind_map

        if self._kind_map is None:
            self._kind_map = dict(kind="ScatterGraph", kwargs=dict())

        # Merge subplots_kwargs
        self._subplots_kwargs = config_subplots_kwargs.copy()
        if subplots_kwargs:
            self._subplots_kwargs.update(subplots_kwargs)

        if not self._subplots_kwargs:  # Empty dict is falsy
            self._init_subplots_kwargs()

        # 计算行列数
        self.__cols = self._subplots_kwargs.get("cols", 1)
        if "rows" in self._subplots_kwargs:
            self.__rows = self._subplots_kwargs["rows"]
        else:
            if self._df is not None:
                self.__rows = math.ceil(len(self._df.columns) / self.__cols)
            else:
                self.__rows = 1  # 默认值

        self._sub_graph_data = sub_graph_data
        # [Fix] 只有在 sub_graph_data 未提供且 df 存在时，才自动生成
        if self._sub_graph_data is None and self._df is not None:
            self._init_sub_graph_data()

        self._grid = None

        # [Fix] 确保有数据可画
        if self._sub_graph_data:
            self._init_figure()

    def _init_sub_graph_data(self):
        """自动生成子图数据配置列表

        根据 DataFrame 的列自动生成子图配置。每列对应一个子图，
        按照指定的列数（cols）自动计算行列位置。

        生成的配置包含:
            - column_name: DataFrame 列名
            - row: 子图所在行号（从 1 开始）
            - col: 子图所在列号（从 1 开始）
            - name: 显示名称（将列名中的下划线替换为空格）
            - kind: 图表类型（从 kind_map 获取）
            - graph_kwargs: 图表参数（从 kind_map 获取）

        .. note::
            此方法仅在 ``sub_graph_data`` 参数未提供且 ``df`` 存在时调用。
            子图按从左到右、从上到下的顺序排列。

        Returns:
            None: 结果存储在 ``self._sub_graph_data`` 中
        """
        self._sub_graph_data = []
        for i, column_name in enumerate(self._df.columns):
            row = math.ceil((i + 1) / self.__cols)
            _temp = (i + 1) % self.__cols
            col = _temp if _temp else self.__cols
            res_name = column_name.replace("_", " ")
            _temp_row_data = (
                column_name,
                dict(
                    row=row,
                    col=col,
                    name=res_name,
                    kind=self._kind_map["kind"],
                    graph_kwargs=self._kind_map["kwargs"],
                ),
            )
            self._sub_graph_data.append(_temp_row_data)

    def _init_subplots_kwargs(self):
        """初始化默认子图布局参数

        当 ``subplots_kwargs`` 参数未提供时，生成默认的布局配置。
        默认配置为单列布局，行数等于 DataFrame 的列数。

        默认参数:
            - rows: DataFrame 列数（每列一行）
            - cols: 1（单列）
            - shared_xaxes: True（共享 X 轴）
            - vertical_spacing: 0.05（垂直间距 5%）
            - row_width: 所有行等高（权重均为 1）

        Returns:
            None: 结果存储在 ``self._subplots_kwargs`` 中
        """
        _cols = 1
        _rows = len(self._df.columns) if self._df is not None else 1
        self._subplots_kwargs = dict()
        self._subplots_kwargs["rows"] = _rows
        self._subplots_kwargs["cols"] = _cols
        self._subplots_kwargs["shared_xaxes"] = True
        self._subplots_kwargs["vertical_spacing"] = 0.05
        self._subplots_kwargs["row_width"] = [1] * _rows

    def _init_figure(self):
        """初始化 Pyecharts Grid 布局并创建所有子图

        该方法执行以下步骤:

        1. **创建 Grid 画布**: 根据 layout 配置创建指定尺寸的 Grid 对象
        2. **计算行列布局**:
           - 根据 row_width 权重计算各行的高度和垂直位置
           - 根据 col_width 权重计算各列的宽度和水平位置
           - 考虑边距（margin_top/bottom/left/right）和间距（spacing）
        3. **分组子图**: 将 sub_graph_data 按 (row, col) 分组
        4. **创建每个子图**:
           - 使用 BaseGraph 工厂方法实例化指定类型的图表
           - 计算标题和图例的绝对位置（相对于整个画布）
           - 应用全局选项覆盖（标题、图例、Tooltip、共享轴等）
           - 将图表添加到 Grid 的指定位置

        .. note::
            - 顶部边距默认为 10% 以容纳标题
            - 多列布局时，标题位置会自动计算以居中显示在各自的 Grid 单元格中
            - 共享 X 轴时，非最后一行的 X 轴标签会被隐藏

        Returns:
            None: 结果存储在 ``self._grid`` 中

        Raises:
            ValueError: 当 Grid 配置参数无效时（如行列数不匹配）
        """
        canvas_height = self._layout.get("height", 1000)
        canvas_width = self._layout.get("width", "100%")
        if isinstance(canvas_width, int):
            canvas_width = f"{canvas_width}px"
        if isinstance(canvas_height, int):
            canvas_height = f"{canvas_height}px"

        self._grid = Grid(
            init_opts=opts.InitOpts(
                width=canvas_width, height=canvas_height, theme=ThemeType.WHITE
            )
        )

        row_width_list = self._subplots_kwargs.get("row_width", [1] * self.__rows)
        if len(row_width_list) < self.__rows:
            row_width_list = [1] * (self.__rows - len(row_width_list)) + row_width_list

        row_weights = list(reversed(row_width_list))
        total_weight = sum(row_weights)
        vertical_spacing = self._subplots_kwargs.get("vertical_spacing", 0.02)

        margin_top_pct = 10  # 增加顶部边距以容纳标题
        margin_bottom_pct = 5
        spacing_pct = vertical_spacing * 100
        available_height_pct = (
            100 - margin_top_pct - margin_bottom_pct - (self.__rows - 1) * spacing_pct
        )

        row_configs = {}
        current_pos = margin_top_pct

        for i, weight in enumerate(row_weights):
            row_idx = i + 1
            h_pct = (weight / total_weight) * available_height_pct
            row_configs[row_idx] = {"pos_top": f"{current_pos}%", "height": f"{h_pct}%"}
            current_pos += h_pct + spacing_pct

        col_width_list = self._subplots_kwargs.get("col_width", [1] * self.__cols)
        if len(col_width_list) < self.__cols:
            col_width_list = col_width_list + [1] * (self.__cols - len(col_width_list))

        total_col_weight = sum(col_width_list)
        horizontal_spacing = self._subplots_kwargs.get("horizontal_spacing", 0.02)
        margin_left_pct = self._subplots_kwargs.get("margin_left", 0.05) * 100
        margin_right_pct = self._subplots_kwargs.get("margin_right", 0.05) * 100
        h_spacing_pct = horizontal_spacing * 100
        available_width_pct = (
            100 - margin_left_pct - margin_right_pct - (self.__cols - 1) * h_spacing_pct
        )

        col_configs = {}
        current_left = margin_left_pct
        for i, weight in enumerate(col_width_list):
            col_idx = i + 1
            w_pct = (weight / total_col_weight) * available_width_pct
            col_configs[col_idx] = {
                "pos_left": f"{current_left}%",
                "width": f"{w_pct}%",
            }
            current_left += w_pct + h_spacing_pct

        cell_groups = {}
        for column_name, column_map in self._sub_graph_data:
            if not isinstance(column_name, str):
                continue
            row = column_map["row"]
            col = column_map.get("col", 1)
            key = (row, col)
            if key not in cell_groups:
                cell_groups[key] = []
            cell_groups[key].append((column_name, column_map))

        shared_xaxes = self._subplots_kwargs.get("shared_xaxes", False)
        # x_axis_data prep removed as each Graph instance handles its own X-axis

        for (row, col), items in cell_groups.items():
            first_item_map = items[0][1]
            kind = first_item_map.get(
                "kind", self._kind_map.get("kind", "ScatterGraph")
            )

            # 1. Prepare Data and Config for the specific cell
            cols_in_cell = [item[0] for item in items]
            cell_df = self._df[cols_in_cell].copy()

            name_dict = {
                col_name: col_map.get("name", col_name.replace("_", " "))
                for col_name, col_map in items
            }

            # Base/Default kwargs
            final_kwargs = self._kind_map.get("kwargs", {}).copy()
            # Merge with cell-specific kwargs
            final_kwargs.update(first_item_map.get("graph_kwargs", {}))

            # 2. Instantiate Graph via Factory
            # This ensures we get the correct Graph type (e.g. DistplotGraph) with all its logic
            graph_instance = BaseGraph.get_instance_with_graph_parameters(
                graph_type=kind,
                df=cell_df,
                name_dict=name_dict,
                graph_kwargs=final_kwargs,
                # Pass layout mostly for title, though Grid handles positioning
                layout={"title": first_item_map.get("title", "")},
            )

            chart = graph_instance.figure

            # 3. Calculate Layout Overrides (Positions)
            title_top_offset = final_kwargs.get("title_top_offset", -4)
            row_cfg = row_configs.get(row, {"pos_top": "50%", "height": "50%"})
            try:
                base_top = float(row_cfg["pos_top"].strip("%"))
                calc_title_top = base_top + title_top_offset
                final_title_top = f"{calc_title_top}%"

                legend_top_offset = final_kwargs.get("legend_top_offset", 0)
                calc_legend_top = base_top + legend_top_offset

                # Check for explicit legend_pos_top override
                explicit_legend_top = final_kwargs.get("legend_pos_top", None)
                if explicit_legend_top is not None:
                    final_legend_top = explicit_legend_top
                else:
                    final_legend_top = f"{calc_legend_top}%"
            except ValueError:
                final_title_top = "auto"
                # Still respect explicit override even if base_top failed
                explicit_legend_top = final_kwargs.get("legend_pos_top", None)
                if explicit_legend_top is not None:
                    final_legend_top = explicit_legend_top
                else:
                    final_legend_top = "auto"

            col_cfg = col_configs.get(col, {"pos_left": "5%", "width": "90%"})

            # Legend Config
            is_show_legend = final_kwargs.get("is_show_legend", True)
            legend_pos_left_custom = final_kwargs.get("legend_pos_left", None)
            legend_pos_right_custom = final_kwargs.get("legend_pos_right", None)

            final_legend_left = (
                legend_pos_left_custom if legend_pos_left_custom is not None else "0%"
            )
            if legend_pos_right_custom is not None:
                final_legend_left = None

            # Subtitle fallback
            sub_title_text = first_item_map.get("title", "")
            if not sub_title_text:
                if row == 1 and col == 1:
                    sub_title_text = self._layout.get("title", "")

            # Identify tooltip formatter from kwargs to re-apply specific style
            tooltip_formatter = BaseGraph._normalize_formatter(
                final_kwargs.get("tooltip_formatter", None)
            )

            # 允许不同图表类型自定义 axis_pointer_type（如 DistplotGraph 使用 "shadow"）
            axis_pointer_type = final_kwargs.get("axis_pointer_type", "line")
            legend_orient = final_kwargs.get("legend_orient", "horizontal")
            legend_data = final_kwargs.get("legend_data", None)

            # Calculate title horizontal position for multi-column layouts
            # Title position in Grid is relative to the entire canvas, not the grid cell
            # So we need to calculate the absolute center position of each grid cell
            if self.__cols > 1:
                # Calculate center position of the current grid cell
                try:
                    cell_left = float(col_cfg["pos_left"].strip("%"))
                    cell_width = float(col_cfg["width"].strip("%"))
                    # Title should be at the center of the grid cell
                    title_pos_left = f"{cell_left + cell_width / 2}%"
                except (ValueError, KeyError):
                    title_pos_left = "center"
            else:
                # Single column: use global layout setting
                title_pos_left = self._layout.get("title_pos_left", "center")

            # 4. Prepare Global Opts Override
            # We explicitly override Title and Legend positions to suit the Grid

            # Construct title configuration as a dict to support 'textAlign' property
            # which is not directly exposed by opts.TitleOpts but supported by ECharts
            title_config = {
                "text": sub_title_text,
                "left": title_pos_left,
                "top": final_title_top,
            }
            # For specific absolute positioning (like centering in a sub-region),
            # we need explicit text alignment.
            if self.__cols > 1:
                title_config["textAlign"] = "center"

            # [Fix] LegendOpts doesn't support 'data' in __init__. We inject it manually.
            _legend_opts = opts.LegendOpts(
                is_show=is_show_legend,
                pos_top=final_legend_top,
                pos_left=final_legend_left,
                pos_right=legend_pos_right_custom,
                orient=legend_orient,
                item_gap=5,
                textstyle_opts=opts.TextStyleOpts(font_size=10),
            )
            if legend_data:
                _legend_opts.opts["data"] = legend_data

            global_opts_updates = {
                "title_opts": title_config,
                "legend_opts": _legend_opts,
                "tooltip_opts": opts.TooltipOpts(
                    trigger="axis",
                    axis_pointer_type=axis_pointer_type,
                    background_color="rgba(255, 255, 255, 0.9)",
                    border_width=1,
                    formatter=tooltip_formatter,
                ),
            }

            # Handle Shared X-Axis (only if enabled and not last row)
            # CAUTION: This replaces xaxis_opts, so use with care on complex graphs
            is_last_row = row == self.__rows
            if shared_xaxes and not is_last_row:
                global_opts_updates["xaxis_opts"] = opts.AxisOpts(
                    axislabel_opts=opts.LabelOpts(is_show=False),
                    splitline_opts=opts.SplitLineOpts(
                        is_show=True,
                        linestyle_opts=opts.LineStyleOpts(opacity=0.5, type_="dashed"),
                    ),
                )

            chart.set_global_opts(**global_opts_updates)

            self._grid.add(
                chart,
                grid_opts=opts.GridOpts(
                    pos_left=col_cfg["pos_left"],
                    pos_right=None,
                    width=col_cfg["width"],
                    pos_top=row_cfg["pos_top"],
                    height=row_cfg["height"],
                    is_contain_label=True,
                ),
            )

    @property
    def figure(self):
        """获取生成的 Pyecharts Grid 对象

        Returns:
            Grid: Pyecharts Grid 对象，包含所有子图。可调用 ``render()`` 或 ``render_notebook()`` 方法进行渲染。

        Examples:
            >>> sg = SubplotsGraph(df=df, config=config)
            >>> grid = sg.figure
            >>> grid.render("output.html")  # 渲染为 HTML 文件
            >>> grid.render_notebook()  # 在 Jupyter Notebook 中显示
        """
        return self._grid
