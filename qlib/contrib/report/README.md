# Qlib Contrib Report 模块重构文档

**文档版本**: v2.0
**更新日期**: 2026-01-29
**作者**: Hugo (shen.lan123@gmail.com)

---

## 📋 目录

- [架构概览](#架构概览)
- [核心重构](#核心重构)
- [新增功能](#新增功能)
- [API 变更](#api-变更)
- [使用指南](#使用指南)
- [迁移指南](#迁移指南)
- [技术细节](#技术细节)

---

## 架构概览

### 目录结构

```
qlib/contrib/report/
├── __init__.py                     # 模块入口
├── graph.py                        # 核心图表类（~1900行）
├── graph_bak.py                    # 备份版本（matplotlib/plotly）
├── display_config.py               # 统一配置管理
├── utils.py                        # 工具函数
│
├── analysis_model/                 # 模型性能分析
│   ├── __init__.py
│   ├── analysis_model_performance.py    # 主分析函数（重构）
│   └── analysis_model_performance_bak.py # 备份版本
│
├── analysis_position/              # 持仓分析
│   ├── __init__.py
│   ├── report.py                   # 报告生成
│   ├── report_bak.py               # 备份版本
│   ├── score_ic.py                 # IC 分析
│   ├── risk_analysis.py            # 风险分析
│   ├── cumulative_return.py        # 累计收益
│   ├── rank_label.py               # 标签排名
│   └── parse_position.py           # 持仓解析
│
├── analysis_alpha/                 # Alpha 因子分析（新增）
│   └── factor_stats.py
│
├── data/                           # 数据处理
│   ├── __init__.py
│   ├── base.py
│   └── ana.py
│
└── docs/                           # 文档
    └── pyecharts_jupyter_guide.md  # Jupyter 环境使用指南
```

### 设计原则

1. **计算与绘图分离**：数据计算逻辑（`compute_*`）与可视化（`_pred_*`）分离
2. **配置集中管理**：通过 `display_config.py` 统一管理所有图表配置
3. **主题一致性**：强制使用白色主题（`ThemeType.WHITE`）确保跨平台显示一致
4. **环境自适应**：自动检测 Jupyter 环境（Notebook/Lab/VS Code）
5. **向后兼容**：保留 `*_bak.py` 备份版本

---

## 核心重构

### 1. 从 Matplotlib/Plotly 迁移到 Pyecharts

**背景**：
- Matplotlib：静态图表，交互性差
- Plotly：交互性好但中文支持弱
- Pyecharts：百度开源，中文友好，交互性强

**迁移范围**：
- ✅ `graph.py` - 所有图表类（1900行）
- ✅ `analysis_model_performance.py` - 模型性能分析
- ✅ `risk_analysis.py` - 风险分析图表
- ✅ `score_ic.py` - IC 分析图表

**保留备份**：
- `graph_bak.py` - 基于 Matplotlib 的原版实现
- `analysis_model_performance_bak.py` - 备份版本
- `report_bak.py` - 备份版本

### 2. 循环导入问题解决

**问题**：
```
display_config.py → graph.py (导入格式化函数)
graph.py → display_config.py (导入配置)
```

**解决方案**：
- ✅ 延迟初始化（工厂函数）
- ✅ `__post_init__` 方法
- ✅ 配置缓存机制（`_cached_configs`）

**示例**：
```python
# 旧版本（循环导入）
from .graph import get_percent_formatter
IC_GRAPH_CONFIG = GraphDisplayConfig(
    tooltip_formatter=JsCode(get_percent_formatter(2))
)

# 新版本（延迟初始化）
def _create_ic_graph_config():
    from .graph import get_percent_formatter
    return GraphDisplayConfig(
        tooltip_formatter=JsCode(get_percent_formatter(2))
    )
IC_GRAPH_CONFIG = _create_ic_graph_config()
```

### 3. 计算与绘图分离

**重构前**：
```python
def _pred_ic(pred_label, ...):
    # 1. 计算 IC
    ic_df = ...
    # 2. 绘制时序图
    graph_ts = ScatterGraph(...)
    # 3. 绘制热力图
    graph_heatmap = HeatmapGraph(...)
    return figs
```

**重构后**：
```python
# 纯计算函数（可复用、可测试）
def compute_ic(pred_label, ...) -> pd.DataFrame:
    """计算 IC/Rank IC"""
    ic_df = ...
    return ic_df

# 绘图函数（调用计算 + 可视化）
def _pred_ic(pred_label, config=None, ...):
    # 1. 计算
    ic_df = compute_ic(pred_label, ...)

    # 2. 可视化
    fig_ts = plot_timeseries(ic_df, config, ...)
    fig_calendar = plot_calendar(ic_df, ...)
    return [fig_ts, fig_calendar, ...]
```

**优势**：
- ✅ 可测试性：计算逻辑可单独测试
- ✅ 可复用性：`compute_*` 函数可被其他模块调用
- ✅ 灵活性：支持自定义 `config` 参数

### 4. 统一图表初始化配置

**问题**：不同图表类初始化参数不统一，导致主题不一致

**解决方案**：新增 `get_default_init_opts()` 函数

```python
def get_default_init_opts(width: str = "100%", height: int = 400) -> opts.InitOpts:
    """获取默认的图表初始化配置

    使用白色主题以确保在 Jupyter Notebook 中正确显示。
    """
    if isinstance(height, int):
        height = f"{height}px"
    return opts.InitOpts(
        width=width,
        height=height,
        theme=ThemeType.WHITE,      # 强制白色主题
        bg_color="#ffffff",
    )
```

**应用到所有图表类**：
- ScatterGraph
- BarGraph
- HeatmapGraph
- HistogramGraph
- QQPlotGraph
- CalendarGraph
- SubplotsGraph (Grid)

---

## 新增功能

### 1. CalendarGraph - 日历热力图

**用途**：以日历形式展示时间序列数据（如日度 IC）

**特性**：
- ✅ 自动检测单年/多年模式
- ✅ 单年：使用 Pyecharts API
- ✅ 多年：使用原生 ECharts JSON 配置（多个 calendar 组件）
- ✅ 紧凑布局优化（节省 31% 垂直空间）

**使用示例**：
```python
from qlib.contrib.report.graph import plot_calendar

# 单年模式
fig = plot_calendar(
    df=ic_df,
    title="Daily IC Calendar",
    layout={"width": "100%", "height": "600px"},
    graph_kwargs={
        "visualMap": {
            "orient": "horizontal",
            "left": "75%",
            "inRange": {
                "color": ["#10b981", "#6ee7b7", "#f3f4f6", "#fca5a5", "#ef4444"]
            }
        }
    }
)
```

**布局优化**（提交 `c4c9c7ca`）：
```
旧版: height = 100 + n * 190
新版: height = 100 + n * 120 + (n-1) * 6 + 20

节省空间:
- 3年: 670px → 482px (28%)
- 5年: 1050px → 728px (31%)
```

### 2. QQPlotGraph - Q-Q 图

**用途**：检验数据是否服从正态分布

**特性**：
- ✅ 使用 `scipy.stats` 计算理论分位数
- ✅ 45 度参考线
- ✅ 自动配置轴标签

**使用示例**：
```python
from qlib.contrib.report.graph import plot_qq
import scipy.stats as stats

# 计算 QQ 图数据
plt_fig = sm.qqplot(ic_data.dropna(), dist=stats.norm, fit=True, line="45")
plt.close(plt_fig)
qqplot_data = plt_fig.gca().lines[0]
df_qq = pd.DataFrame({"Sample Quantiles": qqplot_data.get_ydata()},
                     index=qqplot_data.get_xdata())

# 绘制
fig = plot_qq(
    series=df_qq.iloc[:, 0],
    title="IC Normal Dist. Q-Q",
    config=IC_QQ_CONFIG
)
```

### 3. 便捷绘图函数

**目的**：简化常用图表的调用方式

**函数列表**：
```python
plot_timeseries(df, config, title, layout, graph_kwargs)
plot_distribution(df, config, title, layout, graph_kwargs)
plot_qq(series, config, title, layout, graph_kwargs)
plot_calendar(df, config, title, layout, graph_kwargs)
```

**使用示例**：
```python
from qlib.contrib.report.graph import plot_timeseries

# 简化调用（替代直接实例化 ScatterGraph）
fig = plot_timeseries(
    df=ic_df,
    config=MODEL_PERFORMANCE_CONFIG,
    title="Information Coefficient (IC)",
    layout={"width": "100%"},
    graph_kwargs={"mode": "lines"}
)
```

### 4. 换手率分析

**新增函数**：`_pred_turnover()`

**用途**：衡量 Alpha 信号的稳定性及潜在交易成本

**公式**：
```
Turnover = 1 - (此期组合与上期组合重合数 / 组合总数)
```

**使用示例**：
```python
from qlib.contrib.report.analysis_model import model_performance_graph

figs = model_performance_graph(
    pred_label=pred_df,
    graph_names=["pred_turnover"],  # 新增
    N=5,
    lag=1
)
```

### 5. 自相关分析增强

**重构前**：
- 单一柱状图（Lag 1-5 的平均值）

**重构后**：
- 时序散点图（每日自相关系数）
- 使用 Spearman Rank Correlation
- 展示时间演化趋势

**对比**：
| 特性 | 旧版本 | 新版本 |
|------|--------|--------|
| 可视化 | 柱状图 | 时序散点图 |
| 粒度 | 单一均值 | 每日数值 |
| 相关方法 | Pearson | Spearman Rank |
| 时间维度 | ❌ | ✅ |

### 6. IC 分析增强

**月度热力图 → 日历热力图**（提交 `9dfc8ea2`）：

| 特性 | 月度热力图 | 日历热力图 |
|------|-----------|-----------|
| 时间粒度 | 月聚合 | 日度 |
| 视觉效果 | 12×N 矩阵 | 日历视图 |
| 信息保留 | ❌ 丢失日内信息 | ✅ 完整保留 |
| 布局 | 标准 | 紧凑优化 |

**配色优化**（提交 `e15b251f`）：
```python
"inRange": {
    # A股审美：负值绿色（跌），正值红色（涨）
    "color": ["#10b981", "#6ee7b7", "#f3f4f6", "#fca5a5", "#ef4444"]
}
```

### 7. Jupyter 环境自动检测

**特性**：
- ✅ 自动识别 Jupyter Notebook / Jupyter Lab / VS Code
- ✅ 自动配置 `CurrentConfig.NOTEBOOK_TYPE`
- ✅ 智能加载 JavaScript

**实现**：
```python
class _JupyterEnvironmentDetector:
    """Jupyter 环境检测器"""

    @classmethod
    def detect_environment(cls) -> str:
        """检测当前运行环境"""
        # 1. 检查是否只安装了 jupyterlab
        # 2. 检查是否同时安装了 jupyterlab 和 notebook
        # 3. 返回: 'jupyter_lab', 'jupyter_notebook', 'unknown'
```

**检测策略**（重要）：
- 如果系统中**只安装了 jupyterlab** → 启用 Lab 模式
- 如果系统中**同时安装了 jupyterlab 和 notebook** → 使用 Notebook 模式（兼容性更好）

---

## API 变更

### 1. 模块级导入

**新增导入**：
```python
from qlib.contrib.report.graph import (
    # 便捷绘图函数
    plot_timeseries,
    plot_distribution,
    plot_qq,
    plot_calendar,

    # 格式化函数
    get_number_formatter,
    get_percent_formatter,
    get_axis_number_formatter,
    get_axis_percent_formatter,
    get_calendar_formatter,

    # 初始化配置
    get_default_init_opts,

    # 图表类
    ScatterGraph,
    BarGraph,
    HeatmapGraph,
    HistogramGraph,
    DistplotGraph,
    QQPlotGraph,
    CalendarGraph,
    SubplotsGraph,
)
```

### 2. 配置对象

**位置**：`qlib/contrib/report/display_config.py`

**预定义配置**：
```python
# 报告默认配置
REPORT_DEFAULT_CONFIG

# IC 分析配置
IC_GRAPH_CONFIG
IC_DIST_CONFIG
IC_QQ_CONFIG
IC_CALENDAR_LAYOUT

# 分组收益配置
GROUP_RETURN_CONFIG
GROUP_RETURN_SUBPLOTS_CONFIG

# 模型性能配置
MODEL_PERFORMANCE_CONFIG

# Score IC 配置
SCORE_IC_CONFIG

# 风险分析配置
RISK_ANALYSIS_CONFIG

# 通用配置
TIMESERIES_CONFIG
AUTOCORR_CONFIG
TURNOVER_CONFIG
DIST_CONFIG
QQ_CONFIG
CALENDAR_CONFIG
```

**自定义配置**：
```python
from qlib.contrib.report.display_config import GraphDisplayConfig, LegendConfig

custom_config = GraphDisplayConfig(
    legend=LegendConfig(
        pos_left="75%",
        pos_top="4%",
        orient="horizontal"
    ),
    tooltip_formatter=JsCode(get_number_formatter(4)),
    axis_formatter=JsCode(get_axis_number_formatter(4)),
    height=500,
)
```

### 3. model_performance_graph 参数变更

**新增参数**：
```python
def model_performance_graph(
    pred_label: pd.DataFrame,
    ...
    config: Optional[GraphDisplayConfig] = None,  # 新增：自定义配置
    **kwargs,
) -> List[object]:
```

**使用示例**：
```python
# 使用默认配置
figs = model_performance_graph(pred_label)

# 使用自定义配置
from qlib.contrib.report.display_config import TIMESERIES_CONFIG
figs = model_performance_graph(
    pred_label,
    config=TIMESERIES_CONFIG  # 覆盖默认配置
)
```

### 4. graph_names 新增选项

**新增**：
```python
graph_names = [
    "group_return",      # 分组收益
    "pred_ic",          # IC 分析
    "pred_autocorr",    # 自相关分析
    "pred_turnover",    # 换手率分析（新增）
]
```

---

## 使用指南

### 1. 模型性能分析

**完整流程**：
```python
from qlib.contrib.report.analysis_model import model_performance_graph

# 1. 准备数据（pred_label: MultiIndex [datetime, instrument], columns: [score, label]）
pred_label = ...

# 2. 生成完整报告（4个图表）
figs = model_performance_graph(
    pred_label=pred_label,
    N=5,                    # 分组数
    reverse=False,          # 是否反转分数
    rank=False,             # 是否使用排名
    graph_names=[
        "group_return",     # 1. 分组累计收益 + 分布
        "pred_ic",         # 2. IC/Rank IC 时序 + 日历热力图 + QQ图
        "pred_autocorr",   # 3. 自相关时序
        "pred_turnover",   # 4. 换手率时序
    ],
    show_notebook=True,     # 在 Jupyter 中显示
)

# 3. 保存图表
for i, fig in enumerate(figs):
    fig.render(f"model_performance_{i}.html")
```

**单独使用某个分析**：
```python
from qlib.contrib.report.analysis_model.analysis_model_performance import (
    compute_group_return,
    compute_ic,
    compute_autocorr,
    compute_turnover,
)

# 计算分组收益
group_cum_ret, dist_data = compute_group_return(pred_label, N=5)

# 计算 IC
ic_df = compute_ic(pred_label, methods=["IC", "Rank IC"])

# 计算自相关
autocorr_df = compute_autocorr(pred_label, lag=1)

# 计算换手率
turnover_df = compute_turnover(pred_label, N=5, lag=1)
```

### 2. 持仓分析

**报告生成**：
```python
from qlib.contrib.report import analysis_position

# 1. 标准报告
report_df = analysis_position.report_graph(
    predictor=pred_df,
    show_notebook=True,
)

# 2. IC 分析
figs = analysis_position.score_ic_graph(
    pred_label,
    show_notebook=True,
)

# 3. 累计收益
figs = analysis_position.cumulative_return_graph(
    returns_df,
    show_notebook=True,
)

# 4. 风险分析
figs = analysis_position.risk_analysis_graph(
    analysis_df,
    report_normal_df,
    show_notebook=True,
)
```

### 3. 自定义图表

**使用便捷函数**：
```python
from qlib.contrib.report.graph import (
    plot_timeseries,
    plot_distribution,
    plot_qq,
    plot_calendar,
)

# 时序图
fig_ts = plot_timeseries(
    df=ic_df,
    title="IC Time Series",
    layout={"width": "100%", "height": 400},
)

# 分布图
fig_dist = plot_distribution(
    df=returns_df,
    title="Return Distribution",
    graph_kwargs={"bin_size": 0.01},
)

# QQ 图
fig_qq = plot_qq(
    series=ic_data,
    title="Normality Test",
)

# 日历热力图
fig_cal = plot_calendar(
    df=ic_df,
    title="Daily IC Calendar",
    graph_kwargs={
        "visualMap": {
            "orient": "horizontal",
            "left": "75%",
        }
    },
)
```

**直接使用图表类**：
```python
from qlib.contrib.report.graph import ScatterGraph
from qlib.contrib.report.display_config import MODEL_PERFORMANCE_CONFIG

# 创建图表
graph = ScatterGraph(
    df=ic_df,
    config=MODEL_PERFORMANCE_CONFIG,
    layout={
        "title": "IC Time Series",
        "width": "100%",
        "height": 400,
    },
    graph_kwargs={
        "mode": "lines",
        "is_show_legend": True,
    },
)

# 获取图表对象
fig = graph.figure

# 在 Jupyter 中显示
from qlib.contrib.report.graph import show_graph_in_notebook
show_graph_in_notebook([fig])
```

### 4. 在 Jupyter 中显示

**自动模式**（推荐）：
```python
from qlib.contrib.report.graph import show_graph_in_notebook

# 自动检测环境（Notebook/Lab/VS Code）
show_graph_in_notebook([fig1, fig2, fig3])
```

**手动配置**（特殊情况）：
```python
# Jupyter Lab
from pyecharts.globals import CurrentConfig, NotebookType
CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_LAB

# Jupyter Notebook
CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_NOTEBOOK

# 然后显示
show_graph_in_notebook([fig])
```

---

## 迁移指南

### 从 Matplotlib/Plotly 迁移到 Pyecharts

**步骤 1：更新导入**
```python
# 旧版本
from qlib.contrib.report.analysis_position import risk_analysis_graph

# 新版本（Pyecharts）
from qlib.contrib.report.analysis_position import risk_analysis_graph  # 函数签名不变
```

**步骤 2：检查图表显示**
- ✅ 确保在 Jupyter 中使用 `show_notebook=True`
- ✅ 或手动调用 `show_graph_in_notebook([fig])`

**步骤 3：处理中文显示**
```python
# Pyecharts 默认支持中文，无需额外配置
# 如果遇到乱码，检查字体设置
```

### 从旧版本配置迁移

**配置对象变更**：
```python
# 旧版本（直接在 graph_kwargs 中配置）
graph_kwargs={
    "is_show_legend": True,
    "legend_pos_left": "75%",
    "tooltip_formatter": JsCode(get_number_formatter(2)),
}

# 新版本（使用 config 对象）
from qlib.contrib.report.display_config import GraphDisplayConfig, LegendConfig

config = GraphDisplayConfig(
    legend=LegendConfig(pos_left="75%"),
    tooltip_formatter=JsCode(get_number_formatter(2)),
)

# 使用 config
graph = ScatterGraph(df=df, config=config)
```

### 处理循环导入错误

**如果遇到循环导入错误**：
```python
# ❌ 错误：在模块顶层导入
from .graph import get_percent_formatter
CONFIG = GraphDisplayConfig(tooltip_formatter=...)

# ✅ 正确：使用延迟初始化
def _create_config():
    from .graph import get_percent_formatter
    return GraphDisplayConfig(tooltip_formatter=...)
CONFIG = _create_config()
```

---

## 技术细节

### 1. 格式化函数

**数字格式化**：
```python
get_number_formatter(decimals=2, use_comma=True)
# 输出: "function(params){return Number(params.value).toFixed(2);}"
```

**百分比格式化**：
```python
get_percent_formatter(decimals=2)
# 输出: "function(params){return (Number(params.value)*100).toFixed(2)+'%';}"
```

**轴格式化**：
```python
get_axis_number_formatter(decimals=2)
# 输出: "function(value){return Number(value).toFixed(2);}"

get_axis_percent_formatter(decimals=2)
# 输出: "function(value){return (value*100).toFixed(2)+'%';}"
```

**日历格式化**（trigger="item"）：
```python
get_calendar_formatter(decimals=4, use_comma=True)
# 输出: "function(params){var date=params.value[0];var val=params.value[1];...}"
```

### 2. JsCode 长度限制

**问题**：formatter 函数过长（>300 字符）会导致 HTML 截断

**解决方案**：
- ✅ 使用单行压缩格式
- ✅ 移除注释和空格
- ✅ 使用 `\u003c` 代替 `<`（HTML 转义）

**示例**：
```python
# ❌ 错误：多行、有注释、过长（606字符）
def get_formatter():
    return """function (params) {
        var res = params[0].name + '<br/>';
        // 注释
        for (var i = 0; i < params.length; i++) {
            ...
        }
        return res;
    }"""

# ✅ 正确：单行、无注释、压缩（253字符）
def get_formatter():
    return f"function(params){{var res=params[0].name+'<br/>';for(var i=0;i<params.length;i++){{...}}return res;}}"
```

### 3. 图表类继承结构

```
BaseGraph (基类)
├── ScatterGraph (散点/折线图)
├── BarGraph (柱状图)
├── HeatmapGraph (热力图)
├── HistogramGraph (直方图)
├── DistplotGraph (分布图: KDE + 直方图)
├── QQPlotGraph (Q-Q 图)
├── CalendarGraph (日历热力图)
└── SubplotsGraph (多子图容器)
```

### 4. Grid 布局参数

**SubplotsGraph 使用**：
```python
from qlib.contrib.report.graph import SubplotsGraph
from qlib.contrib.report.display_config import GROUP_RETURN_SUBPLOTS_CONFIG

grid = SubplotsGraph(
    df=dist_data,
    config=GROUP_RETURN_SUBPLOTS_CONFIG,
    sub_graph_data=[
        ("Long-Short", dict(row=1, col=1, name="Long-Short")),
        ("Long-Average", dict(row=1, col=2, name="Long-Average")),
    ],
)
```

**Grid.add() 参数**：
```python
Grid().add(
    chart,                    # 图表对象
    grid_opts=opts.GridOpts(
        pos_left="5%",        # 左边距
        pos_right="55%",      # 右边距
        pos_top="15%",        # 上边距
    ),
)
```

### 5. 配置缓存机制

**实现**：
```python
_cached_configs = {}

def _get_or_create_config(config_name, factory_func):
    """获取或创建配置对象（带缓存）"""
    if config_name not in _cached_configs:
        _cached_configs[config_name] = factory_func()
    return _cached_configs[config_name]

# 使用
IC_GRAPH_CONFIG = _get_or_create_config("ic_graph", _create_ic_graph_config)
```

**优势**：
- ✅ 避免重复创建相同配置
- ✅ 提升模块加载性能
- ✅ 保持配置单例性

---

## 版本历史

### v2.0 (2026-01-29) - 架构重构版本

**重大变更**：
- ✅ 解决循环导入问题（延迟初始化）
- ✅ 统一图表初始化配置（`get_default_init_opts`）
- ✅ 计算与绘图分离（`compute_*` 函数）
- ✅ 新增便捷绘图函数（`plot_*`）

**新增功能**：
- ✅ CalendarGraph（日历热力图）
- ✅ 换手率分析（`_pred_turnover`）
- ✅ 自相关分析增强（时序图）
- ✅ IC 日历热力图（替代月度热力图）

**提交记录**：
- `78e2bfef` refactor: 解决循环导入问题并统一图表初始化配置
- `eeddfe6a` refactor: 重构模型性能分析模块实现计算与绘图分离
- `e15b251f` feat: 增强模型性能分析功能并优化IC热力图配色
- `9dfc8ea2` feat: 将月度IC热力图升级为日历热力图并新增CalendarGraph类
- `c4c9c7ca` refactor: 优化多年Calendar热力图垂直布局提升空间利用率
- `4826e70f` refactor: 优化分组收益图图例位置以提升图表区域利用率

### v1.0 (2026-01-22) - Pyecharts 迁移版本

**核心变更**：
- ✅ 从 Matplotlib/Plotly 迁移到 Pyecharts
- ✅ Jupyter 环境自动检测
- ✅ 标题居中显示
- ✅ 中文支持增强

**提交记录**：
- `f2420c52` feat: 新增 display_config 配置模块
- `6bafd347` feat: 重构模型性能分析图表使用 pyecharts
- `f979e09d` feat: 优化风险分析图表
- `c3b09143` feat: 重构模型性能分析图表
- `6a17111a` feat: 添加 Jupyter 环境自动检测

---

## 常见问题

### Q1: 图表在 Jupyter Lab 中显示空白？

**解决方案**：
```python
from pyecharts.globals import CurrentConfig, NotebookType
CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_LAB
```

### Q2: 如何自定义图表颜色？

**方法 1**：通过 `series_colors`：
```python
from qlib.contrib.report.display_config import GraphDisplayConfig

config = GraphDisplayConfig(
    series_colors={
        "Rank IC": "#f0811e",
        "IC": "#1f77b4",
    }
)
```

**方法 2**：通过 `visualMap`（热力图）：
```python
graph_kwargs={
    "visualMap": {
        "inRange": {
            "color": ["#10b981", "#6ee7b7", "#f3f4f6", "#fca5a5", "#ef4444"]
        }
    }
}
```

### Q3: 如何调整图例位置？

**方法**：修改 `LegendConfig`：
```python
from qlib.contrib.report.display_config import LegendConfig

config = GraphDisplayConfig(
    legend=LegendConfig(
        pos_left="75%",      # 靠右
        pos_top="4%",
        orient="horizontal",  # 水平排列
    )
)
```

### Q4: 如何保存图表？

**方法 1**：保存为 HTML
```python
fig.render("output.html")
```

**方法 2**：保存为 PNG（需要 snapshot-engine）
```python
fig.render("output.png", driver="chrome")
```

### Q5: 如何批量处理多个图表？

**使用循环**：
```python
figs = model_performance_graph(pred_label)

for i, fig in enumerate(figs):
    fig.render(f"report_{i}.html")
```

**使用 Grid 组合**：
```python
from pyecharts.charts import Grid

page = Grid()
for fig in figs:
    page.add(fig)

page.render("combined_report.html")
```

---

## 参考资源

### 官方文档
- [Pyecharts 官方文档](https://github.com/pyecharts/pyecharts)
- [ECharts 配置项手册](https://echarts.apache.org/zh/option.html)
- [QLib 官方文档](https://qlib.readthedocs.io/)

### 内部文档
- [Jupyter 环境使用指南](docs/pyecharts_jupyter_guide.md)
- [Pyecharts JsCode 使用规范](../CLAUDE.md#pyecharts-jscode-使用规范)

### 相关 Issues
- [pyecharts #1756 - JupyterLab 显示空白](https://github.com/pyecharts/pyecharts/issues/1756)

---

**文档维护**: Hugo (shen.lan123@gmail.com)
**最后更新**: 2026-01-29
