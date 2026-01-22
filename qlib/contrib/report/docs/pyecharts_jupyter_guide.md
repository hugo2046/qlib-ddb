# Pyecharts 图表在 Jupyter 环境中的显示指南

## 概述

`qlib.contrib.report.graph` 模块已经使用 pyecharts 重构,提供现代化的交互式图表。本文档说明如何在不同的 Jupyter 环境中正确显示图表。

## 最新更新 (2026-01-22)

### ✨ 新功能

1. **自动环境检测**: 自动识别 Jupyter Notebook / Jupyter Lab / VS Code
2. **标题居中显示**: 图表标题默认居中对齐 (`pos_left="center"`)
3. **零配置使用**: 无需手动设置 `CurrentConfig.NOTEBOOK_TYPE`

### 支持的环境

- ✅ **Jupyter Notebook** (7.x 及以下版本)
- ✅ **Jupyter Lab** (3.x 及以上版本)
- ✅ **Classic Notebook**
- ✅ **VS Code Jupyter**

### 自动配置

`show_graph_in_notebook()` 方法会自动:
1. 检测当前的 Jupyter 环境类型
2. 对于 Jupyter Lab,自动配置 `CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_LAB`
3. 智能处理 JavaScript 加载
4. 标题默认居中对齐

**检测策略** (重要):
- 如果系统中**只安装了 jupyterlab** (没有 notebook),自动启用 Lab 模式
- 如果系统中**同时安装了 jupyterlab 和 notebook**,默认使用 Notebook 模式 (兼容性更好)
- 这样可以确保现有 Jupyter Notebook 环境不受影响

**用户无需手动配置任何参数!**

---

## 使用方法

### 基础用法

```python
from qlib.contrib.report.graph import show_graph_in_notebook
from pyecharts.charts import Line, Bar

# 创建图表
line = Line()
line.add_xaxis(["A", "B", "C"])
line.add_yaxis("Series 1", [1, 2, 3])

bar = Bar()
bar.add_xaxis(["A", "B", "C"])
bar.add_yaxis("Series 1", [10, 20, 30])

# 显示图表 (自动适配环境)
show_graph_in_notebook([line, bar])
```

### 在 Jupyter Notebook 中

**直接使用即可,无需额外配置:**

```python
# Cell 1: 导入和创建图表
from qlib.contrib.report.graph import show_graph_in_notebook
from pyecharts.charts import Line

line = Line()
line.add_xaxis(["A", "B", "C"])
line.add_yaxis("Series 1", [1, 2, 3])

# Cell 2: 显示图表
show_graph_in_notebook([line])
```

### 在 Jupyter Lab 中

**同样无需额外配置,系统会自动适配:**

```python
# Cell 1: 导入和创建图表
from qlib.contrib.report.graph import show_graph_in_notebook
from pyecharts.charts import Line

line = Line()
line.add_xaxis(["A", "B", "C"])
line.add_yaxis("Series 1", [1, 2, 3])

# Cell 2: 显示图表 (自动配置 CurrentConfig.NOTEBOOK_TYPE)
show_graph_in_notebook([line])
```

---

## 技术实现

### 环境检测逻辑

```python
class _JupyterEnvironmentDetector:
    """内部类: Jupyter 环境检测器"""

    @classmethod
    def _is_jupyter_lab(cls) -> bool:
        """检测是否在 Jupyter Lab 环境中"""
        try:
            import jupyterlab
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython is not None and hasattr(ipython, 'kernel'):
                return True
        except ImportError:
            pass
        return False

    @classmethod
    def configure_pyecharts(cls):
        """配置 pyecharts 以适配当前的 Jupyter 环境"""
        env_type = cls.detect_environment()

        if env_type == 'jupyter_lab':
            from pyecharts.globals import CurrentConfig, NotebookType
            CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_LAB
```

### show_graph_in_notebook 实现要点

```python
@staticmethod
def show_graph_in_notebook(figure_list: Iterable = None):
    # 1. 配置环境 (只执行一次)
    _JupyterEnvironmentDetector.configure_pyecharts()

    # 2. 尝试加载 JavaScript (对于 Jupyter Lab)
    _JupyterEnvironmentDetector.load_javascript_if_needed()

    # 3. 渲染图表
    for _chart in figure_list:
        if hasattr(_chart, "render_notebook"):
            from IPython.display import display as ipy_display
            rendered = _chart.render_notebook()
            ipy_display(rendered)
```

---

## 标题自定义

### 默认居中

从 **2026-01-22** 开始,所有图表标题默认居中对齐 (`pos_left="center"`)。

### 自定义标题位置

如果需要自定义标题位置,可以在 `layout` 参数中指定:

```python
from qlib.contrib.report.graph import SubplotsGraph

# 自定义标题位置
layout = {
    "height": 1200,
    "width": "100%",
    "title": "我的报告",
    "title_pos_left": "5%",  # 标题靠左
    # 其他选项: "center", "right", 或像素值 "100px"
}

grid = SubplotsGraph(
    df=report_df,
    layout=layout,
    # ...
).figure
```

### 标题位置选项

`title_pos_left` 支持以下值:
- `"center"` (默认): 居中对齐
- `"left"` 或 `"5%"`: 靠左对齐
- `"right"` 或 `"95%"`: 靠右对齐
- 百分比值: `"10%"`, `"20%"`, 等
- 像素值: `"100px"`, `"200px"`, 等

---

## 常见问题

### Q1: 系统同时安装了 Jupyter Lab 和 Notebook,如何在 Lab 中使用?

**问题**: 我的系统同时安装了 jupyterlab 和 notebook,在 Jupyter Lab 中图表显示不正常。

**解决方案**: 手动配置为 Lab 模式

```python
# 在 Notebook 的第一个 cell 中配置
from pyecharts.globals import CurrentConfig, NotebookType
CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_LAB

# 然后正常使用
from qlib.contrib.report import analysis_position
analysis_position.report_graph(report_df, show_notebook=True)
```

### Q2: 在 Jupyter Lab 中图表显示空白?

**原因**: 首次渲染时 JavaScript 未完全加载。

**解决方案**:
1. **方法 1 (推荐)**: 确保已配置为 Lab 模式 (见 Q1)
2. **方法 2**: 手动调用 `load_javascript()`

```python
# Cell 1: 加载 JavaScript (仅 Jupyter Lab 需要)
from pyecharts.charts import Line
Line.load_javascript()

# Cell 2: 渲染图表
line = Line()
# ... 配置图表 ...
line.render_notebook()
```

### Q3: 在 Jupyter Notebook 中图表无法显示?

**原因**: 自动检测错误地配置为 Lab 模式。

**解决方案**: 强制使用 Notebook 模式

```python
# 在 Notebook 的第一个 cell 中配置
from pyecharts.globals import CurrentConfig, NotebookType
CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_NOTEBOOK

# 然后正常使用
from qlib.contrib.report import analysis_position
analysis_position.report_graph(report_df, show_notebook=True)
```

### Q4: 如何确认当前环境类型?

```python
from qlib.contrib.report.graph import _JupyterEnvironmentDetector

env = _JupyterEnvironmentDetector.detect_environment()
print(f"当前环境: {env}")
# 输出: 'jupyter_notebook' 或 'jupyter_lab' 或 'unknown'
```

### Q5: 图表显示不完整或样式错乱?

**原因**: 可能是 `CurrentConfig.NOTEBOOK_TYPE` 配置错误。

**解决方案**: 根据实际环境手动配置 (参考 Q1 和 Q3)

### Q6: VS Code Jupyter 中图表不显示?

**解决方案**: VS Code Jupyter 默认使用 Jupyter Lab 协议,应该能自动适配。如果不显示:

```python
# 尝试显式配置
from pyecharts.globals import CurrentConfig, NotebookType
CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_LAB

# 然后正常显示
show_graph_in_notebook([your_chart])
```

---

## 高级用法

### 单独使用某个图表类

```python
from qlib.contrib.report.graph import show_graph_in_notebook
from qlib.contrib.report.graph import BaseGraph

# 创建自定义图表
graph = BaseGraph.create_graph(graph_class_name="LineGraph")

# 显示
show_graph_in_notebook([graph.figure])
```

### 批量显示多个图表

```python
from qlib.contrib.report.graph import show_graph_in_notebook

# 假设你有多个图表对象
charts = [chart1, chart2, chart3, ...]

# 一次性显示
show_graph_in_notebook(charts)
```

---

## 参考资源

- [pyecharts GitHub Issue #1756 - JupyterLab 显示空白问题](https://github.com/pyecharts/pyecharts/issues/1756)
- [Pyecharts 在 JupyterLab 中无法显示的解决方案](https://blog.csdn.net/silent1cat/article/details/117944987)
- [Pyecharts 无法在jupyterlab中显示问题](https://zhuanlan.zhihu.com/p/578417525)
- [Pyecharts 官方文档](https://github.com/pyecharts/pyecharts)

---

## 版本历史

- **v1.2** (2026-01-22):
  - 🐛 **修复**: 修正环境检测逻辑,避免误将 Notebook 识别为 Lab
  - 📝 **策略**: 如果同时安装 jupyterlab 和 notebook,默认使用 Notebook 模式
  - 📝 **文档**: 添加手动配置说明 (Q1-Q3)

- **v1.1** (2026-01-22):
  - ✨ **新功能**: 图表标题默认居中对齐
  - 📝 **文档**: 添加标题自定义说明

- **v1.0** (2026-01-22):
  - 添加自动环境检测功能
  - 支持 Jupyter Notebook 和 Jupyter Lab 自动适配
  - 用户无需手动配置 `CurrentConfig.NOTEBOOK_TYPE`

---

**生成时间**: 2026-01-22
**作者**: Hugo (shen.lan123@gmail.com)
