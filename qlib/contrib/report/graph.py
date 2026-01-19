'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2026-01-16 16:57:25
LastEditors: shen.lan123@gmail.com
LastEditTime: 2026-01-19 16:52:53
Description: pyecharts重构画图
'''
import math
import importlib
import numpy as np
import pandas as pd
from typing import Iterable, List, Any

# 引入 Pyecharts 组件
from pyecharts.charts import Line, Bar, HeatMap, Grid
from pyecharts import options as opts
from pyecharts.globals import ThemeType

class BaseGraph:
    _name = None

    def __init__(
        self, 
        df: pd.DataFrame = None, 
        layout: dict = None, 
        graph_kwargs: dict = None, 
        name_dict: dict = None, 
        **kwargs
    ):
        self._df = df
        self._layout = dict() if layout is None else layout
        self._graph_kwargs = dict() if graph_kwargs is None else graph_kwargs
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
        for _chart in figure_list:
            if hasattr(_chart, "render_notebook"):
                try:
                    from IPython.display import display
                    display(_chart.render_notebook())
                except ImportError:
                    print("IPython not found, cannot render in notebook.")

    def _apply_global_opts(self):
        if not self.chart:
            return

        # 1. 获取基础配置
        is_show_legend = self._graph_kwargs.get("is_show_legend", True)
        legend_pos_top = self._graph_kwargs.get("legend_pos_top", "5%")
        legend_pos_left = self._graph_kwargs.get("legend_pos_left", None)
        legend_pos_right = self._graph_kwargs.get("legend_pos_right", None)
        legend_orient = self._graph_kwargs.get("legend_orient", "horizontal")
        
        if legend_pos_right is not None:
            legend_pos_left = None

        axis_formatter = self._graph_kwargs.get("axis_formatter", None)
        
        yaxis_opts = opts.AxisOpts(
            is_scale=True,
            axislabel_opts=opts.LabelOpts(formatter=axis_formatter),
            splitline_opts=opts.SplitLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(opacity=0.5, type_="dashed"))
        )

        # 2. 应用配置 (支持 title_pos_left 和 orient)
        self.chart.set_global_opts(
            title_opts=opts.TitleOpts(
                title=self._layout.get("title", ""),
                pos_left=self._layout.get("title_pos_left", "auto") # <--- 支持标题位置配置
            ),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="line"),
            legend_opts=opts.LegendOpts(
                is_show=is_show_legend,
                pos_top=legend_pos_top,
                pos_left=legend_pos_left,
                pos_right=legend_pos_right,
                orient=legend_orient, # <--- 支持图例方向配置
            ),
            xaxis_opts=opts.AxisOpts(type_="category", is_scale=True),
            yaxis_opts=yaxis_opts
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
            
            self.chart.add_yaxis(
                series_name=final_name,
                y_axis=y_data,
                is_symbol_show=is_symbol_show,
                areastyle_opts=areastyle_opts,
                markarea_opts=markarea_opts,
                label_opts=opts.LabelOpts(is_show=False),
                is_smooth=False,
            )


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
            
            self.chart.add_yaxis(
                series_name=final_name,
                y_axis=y_data,
                label_opts=opts.LabelOpts(is_show=False)
            )


class DistplotGraph(BaseGraph):
    _name = "distplot"
    def _init_chart(self):
        self.chart = Bar()
        _t_df = self._df.dropna()
        for col, name in self._name_dict.items():
            data = _t_df[col].values
            hist, bin_edges = np.histogram(data, bins=50)
            x_data = [f"{e:.2f}" for e in bin_edges[:-1]]
            if not self.chart.options.get("xAxis"):
                 self.chart.add_xaxis(x_data)
            self.chart.add_yaxis(
                series_name=name,
                y_axis=hist.tolist(),
                category_gap=0,
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(opacity=0.6)
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
            value=[[i, j, float(self._df.iloc[j, i])] for i in range(len(x_axis)) for j in range(len(y_axis))],
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
            y_data = [x * 100 if not pd.isna(x) else None for x in self._df[col].tolist()]
            self.chart.add_yaxis(
                series_name=f"{name} (%)",
                y_axis=y_data,
                category_gap=0,
                label_opts=opts.LabelOpts(is_show=False)
            )

class SubplotsGraph:
    """
    类似于 df.plot(subplots=True) 的子图管理器
    使用 Pyecharts Grid 实现
    """

    def __init__(
        self,
        df: pd.DataFrame = None,
        kind_map: dict = None,
        layout: dict = None,
        sub_graph_layout: dict = None,
        sub_graph_data: list = None,
        subplots_kwargs: dict = None,
        **kwargs,
    ):
        self._df = df
        self._layout = layout or {}
        self._sub_graph_layout = sub_graph_layout or {}

        self._kind_map = kind_map
        if self._kind_map is None:
            self._kind_map = dict(kind="ScatterGraph", kwargs=dict())

        self._subplots_kwargs = subplots_kwargs
        if self._subplots_kwargs is None:
            self._init_subplots_kwargs()

        # 计算行列数
        self.__cols = self._subplots_kwargs.get("cols", 1)
        if "rows" in self._subplots_kwargs:
            self.__rows = self._subplots_kwargs["rows"]
        else:
            if self._df is not None:
                self.__rows = math.ceil(len(self._df.columns) / self.__cols)
            else:
                self.__rows = 1 # 默认值

        self._sub_graph_data = sub_graph_data
        # [Fix] 只有在 sub_graph_data 未提供且 df 存在时，才自动生成
        if self._sub_graph_data is None and self._df is not None:
            self._init_sub_graph_data()

        self._grid = None
        
        # [Fix] 确保有数据可画
        if self._sub_graph_data:
            self._init_figure()

    def _init_sub_graph_data(self):
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
        _cols = 1
        _rows = len(self._df.columns) if self._df is not None else 1
        self._subplots_kwargs = dict()
        self._subplots_kwargs["rows"] = _rows
        self._subplots_kwargs["cols"] = _cols
        self._subplots_kwargs["shared_xaxes"] = True
        self._subplots_kwargs["vertical_spacing"] = 0.05
        self._subplots_kwargs["row_width"] = [1] * _rows

    def _init_figure(self):
        canvas_height = self._layout.get("height", 1000)
        canvas_width = self._layout.get("width", "100%")
        if isinstance(canvas_width, int): canvas_width = f"{canvas_width}px"
        if isinstance(canvas_height, int): canvas_height = f"{canvas_height}px"

        self._grid = Grid(
            init_opts=opts.InitOpts(
                width=canvas_width, 
                height=canvas_height,
                theme=ThemeType.WHITE
            )
        )

        row_width_list = self._subplots_kwargs.get("row_width", [1] * self.__rows)
        if len(row_width_list) < self.__rows:
             row_width_list = [1] * (self.__rows - len(row_width_list)) + row_width_list
        
        row_weights = list(reversed(row_width_list))
        total_weight = sum(row_weights)
        vertical_spacing = self._subplots_kwargs.get("vertical_spacing", 0.02)
        
        margin_top_pct = 5
        margin_bottom_pct = 5
        spacing_pct = vertical_spacing * 100
        available_height_pct = 100 - margin_top_pct - margin_bottom_pct - (self.__rows - 1) * spacing_pct
        
        row_configs = {}
        current_pos = margin_top_pct
        
        for i, weight in enumerate(row_weights):
            row_idx = i + 1
            h_pct = (weight / total_weight) * available_height_pct
            row_configs[row_idx] = {
                "pos_top": f"{current_pos}%",
                "height": f"{h_pct}%"
            }
            current_pos += h_pct + spacing_pct

        col_width_list = self._subplots_kwargs.get("col_width", [1] * self.__cols)
        if len(col_width_list) < self.__cols:
            col_width_list = col_width_list + [1] * (self.__cols - len(col_width_list))
        
        total_col_weight = sum(col_width_list)
        horizontal_spacing = self._subplots_kwargs.get("horizontal_spacing", 0.02)
        margin_left_pct = 5
        margin_right_pct = 5
        h_spacing_pct = horizontal_spacing * 100
        available_width_pct = 100 - margin_left_pct - margin_right_pct - (self.__cols - 1) * h_spacing_pct
        
        col_configs = {}
        current_left = margin_left_pct
        for i, weight in enumerate(col_width_list):
            col_idx = i + 1
            w_pct = (weight / total_col_weight) * available_width_pct
            col_configs[col_idx] = {
                "pos_left": f"{current_left}%",
                "width": f"{w_pct}%"
            }
            current_left += w_pct + h_spacing_pct

        cell_groups = {}
        for column_name, column_map in self._sub_graph_data:
            if not isinstance(column_name, str): continue
            row = column_map["row"]
            col = column_map.get("col", 1)
            key = (row, col)
            if key not in cell_groups: cell_groups[key] = []
            cell_groups[key].append((column_name, column_map))

        shared_xaxes = self._subplots_kwargs.get("shared_xaxes", False)
        x_axis_data = self._df.index.astype(str).tolist()

        for (row, col), items in cell_groups.items():
            first_item_map = items[0][1]
            kind = first_item_map.get("kind", self._kind_map.get("kind", "ScatterGraph"))
            
            chart = Bar() if "Bar" in kind else Line()
            
            chart.add_xaxis(x_axis_data)

            # 获取参数
            final_kwargs_sample = self._kind_map.get("kwargs", {}).copy()
            final_kwargs_sample.update(first_item_map.get("graph_kwargs", {}))
            
            is_xy_reverse = "Bar" in kind and final_kwargs_sample.get("xy_reverse", False)
            axis_formatter = final_kwargs_sample.get("axis_formatter", None)
            
            # 控制图例
            is_show_legend = final_kwargs_sample.get("is_show_legend", True)
            legend_pos_left_custom = final_kwargs_sample.get("legend_pos_left", None)
            legend_pos_right_custom = final_kwargs_sample.get("legend_pos_right", None)
            legend_pos_top_custom = final_kwargs_sample.get("legend_pos_top", None)
            
            # [Fix 1: Title Position]
            # 动态计算 Title 的位置，使其位于子图上方，而不是默认的 top="auto"
            title_top_offset = final_kwargs_sample.get("title_top_offset", -4) # 默认向上偏移 4%
            
            row_cfg = row_configs.get(row, {"pos_top": "50%", "height": "50%"})
            try:
                # 解析 "15.5%" -> 15.5
                base_top = float(row_cfg["pos_top"].strip('%'))
                
                # 计算 Title Top (向上偏移)
                calc_title_top = base_top + title_top_offset
                final_title_top = f"{calc_title_top}%"
                
                # 计算 Legend Top (向下偏移，进入网格)
                legend_top_offset = final_kwargs_sample.get("legend_top_offset", 0) 
                calc_legend_top = base_top + legend_top_offset
                final_legend_top = f"{calc_legend_top}%"
                
            except ValueError:
                final_title_top = "auto"
                final_legend_top = "auto"

            # 柱宽
            bar_max_width = final_kwargs_sample.get("bar_max_width", None)

            if is_xy_reverse:
                chart.reversal_axis()
            
            for col_name, col_map in items:
                series_name = col_map.get("name", col_name.replace("_", " "))
                
                raw_data = self._df[col_name].tolist()
                # [Pure] 仅负责处理 NaN
                y_data = [x if pd.notna(x) else None for x in raw_data]
                
                final_series_name = series_name

                final_kwargs = self._kind_map.get("kwargs", {}).copy()
                final_kwargs.update(col_map.get("graph_kwargs", {}))
                
                areastyle_opts = None
                if final_kwargs.get("fill") == "tozeroy":
                    areastyle_opts = opts.AreaStyleOpts(opacity=0.3)
                
                mode = final_kwargs.get("mode", "lines")
                is_symbol_show = "markers" in mode
                markarea_opts = final_kwargs.get("markarea_opts", None)
                
                if isinstance(chart, Line):
                    chart.add_yaxis(
                        series_name=final_series_name,
                        y_axis=y_data,
                        is_symbol_show=is_symbol_show,
                        areastyle_opts=areastyle_opts,
                        markarea_opts=markarea_opts,
                        label_opts=opts.LabelOpts(is_show=False),
                        is_smooth=False,
                    )
                else:
                    chart.add_yaxis(
                        series_name=final_series_name,
                        y_axis=y_data,
                        label_opts=opts.LabelOpts(is_show=False),
                        bar_max_width=bar_max_width, 
                    )

            col_cfg = col_configs.get(col, {"pos_left": "5%", "width": "90%"})

            is_last_row = (row == self.__rows)
            xaxis_show_label = True
            if shared_xaxes and not is_last_row:
                xaxis_show_label = False
            
            final_legend_left = legend_pos_left_custom if legend_pos_left_custom is not None else col_cfg["pos_left"]
            if legend_pos_right_custom is not None:
                final_legend_left = None
            
            if is_xy_reverse:
                xaxis_config = opts.AxisOpts(
                    type_="value",
                    axislabel_opts=opts.LabelOpts(formatter=axis_formatter),
                    splitline_opts=opts.SplitLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(opacity=0.5, type_="dashed"))
                )
                yaxis_config = opts.AxisOpts(
                    type_="category",
                    boundary_gap=True,
                    is_inverse=True,
                    axislabel_opts=opts.LabelOpts(is_show=True), 
                    axistick_opts=opts.AxisTickOpts(is_show=False),
                )
            else:
                xaxis_config = opts.AxisOpts(
                    type_="category",
                    boundary_gap=False if isinstance(chart, Line) else True,
                    axislabel_opts=opts.LabelOpts(is_show=xaxis_show_label),
                    axistick_opts=opts.AxisTickOpts(is_show=xaxis_show_label),
                )
                yaxis_config = opts.AxisOpts(
                    is_scale=True,
                    axislabel_opts=opts.LabelOpts(formatter=axis_formatter),
                    splitline_opts=opts.SplitLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(opacity=0.5, type_="dashed"))
                )

            # 获取子图标题
            sub_title_text = first_item_map.get("title", "")
            if not sub_title_text:
                 if row == 1 and col == 1:
                     sub_title_text = self._layout.get("title", "")

            chart.set_global_opts(
                title_opts=opts.TitleOpts(
                    title=sub_title_text,
                    pos_left=self._layout.get("title_pos_left", "auto"),
                    pos_top=final_title_top, # [Fix] 设置标题位置，防止堆叠
                ),
                legend_opts=opts.LegendOpts(
                    is_show=is_show_legend, 
                    pos_top=final_legend_top, # [Fix] 设置图例位置
                    pos_left=final_legend_left,
                    pos_right=legend_pos_right_custom,
                    orient="horizontal",
                    item_gap=5,
                    textstyle_opts=opts.TextStyleOpts(font_size=10)
                ),
                tooltip_opts=opts.TooltipOpts(
                    trigger="axis", 
                    axis_pointer_type="line", 
                    background_color="rgba(255, 255, 255, 0.9)",
                    border_width=1
                ),
                xaxis_opts=xaxis_config,
                yaxis_opts=yaxis_config
            )

            self._grid.add(
                chart,
                grid_opts=opts.GridOpts(
                    pos_left=col_cfg["pos_left"],
                    pos_right=None,
                    width=col_cfg["width"],
                    pos_top=row_cfg["pos_top"],
                    height=row_cfg["height"],
                    is_contain_label=True
                )
            )

    @property
    def figure(self):
        return self._grid