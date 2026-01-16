'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2026-01-16 16:57:25
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2026-01-17 00:33:37
Description: pyecharts重构画图
'''


import math
import importlib
import numpy as np
import pandas as pd
from typing import Iterable, List

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

        self.chart.set_global_opts(
            title_opts=opts.TitleOpts(title=self._layout.get("title", "")),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
            xaxis_opts=opts.AxisOpts(type_="category", is_scale=True),
            yaxis_opts=opts.AxisOpts(is_scale=True)
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
            # [Fix] 1. 后端数据缩放：*100
            raw_data = self._df[col].tolist()
            y_data = []
            for x in raw_data:
                if pd.isna(x):
                    y_data.append(None)
                else:
                    y_data.append(round(x * 100, 2))
            
            # [Fix] 2. 系列名增加 (%) 后缀，这样 Tooltip 显示时会有单位
            final_name = f"{name} (%)"

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

        for col, name in self._name_dict.items():
            raw_data = self._df[col].tolist()
            y_data = [x * 100 if not pd.isna(x) else None for x in raw_data]
            final_name = f"{name} (%)"
            
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
            self.__rows = math.ceil(len(self._df.columns) / self.__cols)

        self._sub_graph_data = sub_graph_data
        if self._sub_graph_data is None:
            self._init_sub_graph_data()

        self._grid = None
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
        _rows = len(self._df.columns)
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
            
            for col_name, col_map in items:
                series_name = col_map.get("name", col_name.replace("_", " "))
                
                # [Fix] 1. 后端数据处理：乘100
                raw_data = self._df[col_name].tolist()
                y_data = []
                for x in raw_data:
                    if pd.isna(x):
                        y_data.append(None)
                    else:
                        y_data.append(round(x * 100, 2))
                
                # [Fix] 2. 修改 Series Name，让 Tooltip 自带单位
                # 这样 Tooltip 会显示 "cum_return (%): 15.23"
                final_series_name = f"{series_name} (%)"

                final_kwargs = self._kind_map.get("kwargs", {}).copy()
                final_kwargs.update(col_map.get("graph_kwargs", {}))
                
                areastyle_opts = None
                if final_kwargs.get("fill") == "tozeroy":
                    areastyle_opts = opts.AreaStyleOpts(opacity=0.3)
                
                mode = final_kwargs.get("mode", "lines")
                is_symbol_show = "markers" in mode
                markarea_opts = final_kwargs.get("markarea_opts", None)
                
                chart.add_yaxis(
                    series_name=final_series_name,
                    y_axis=y_data,
                    is_symbol_show=is_symbol_show,
                    areastyle_opts=areastyle_opts,
                    markarea_opts=markarea_opts,
                    label_opts=opts.LabelOpts(is_show=False),
                    is_smooth=False,
                )

            layout_cfg = row_configs.get(row, {"pos_top": "50%", "height": "50%"})
            is_last_row = (row == self.__rows)
            xaxis_show_label = True
            if shared_xaxes and not is_last_row:
                xaxis_show_label = False
            
            legend_pos_top = layout_cfg["pos_top"]
            
            chart.set_global_opts(
                title_opts=opts.TitleOpts(
                    title=self._layout.get("title", "") if row == 1 else "",
                    pos_left="center"
                ),
                legend_opts=opts.LegendOpts(
                    is_show=True, 
                    pos_top=legend_pos_top, 
                    pos_left="6%",
                    orient="horizontal",
                    item_gap=15,
                    textstyle_opts=opts.TextStyleOpts(font_size=11)
                ),
                # [Fix] 3. 移除 JS Formatter，回归原生
                tooltip_opts=opts.TooltipOpts(
                    trigger="axis", 
                    axis_pointer_type="line",
                    background_color="rgba(255, 255, 255, 0.9)",
                    border_width=1
                ),
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    boundary_gap=False,
                    axislabel_opts=opts.LabelOpts(is_show=xaxis_show_label),
                    axistick_opts=opts.AxisTickOpts(is_show=xaxis_show_label),
                ),
                yaxis_opts=opts.AxisOpts(
                    is_scale=True,
                    # [Fix] 4. Y 轴使用纯字符串模板 (无需 JS)
                    # {value} 会自动替换为数值，" %" 是纯文本
                    axislabel_opts=opts.LabelOpts(formatter="{value} %"),
                    splitline_opts=opts.SplitLineOpts(
                        is_show=True, 
                        linestyle_opts=opts.LineStyleOpts(opacity=0.5, type_="dashed")
                    )
                )
            )

            self._grid.add(
                chart,
                grid_opts=opts.GridOpts(
                    pos_left="5%",
                    pos_right="5%",
                    pos_top=layout_cfg["pos_top"],
                    height=layout_cfg["height"]
                )
            )

    @property
    def figure(self):
        return self._grid