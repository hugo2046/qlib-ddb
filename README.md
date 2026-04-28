[![Python Versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/pyqlib/#files)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos-lightgrey)](https://pypi.org/project/pyqlib/#files)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

# Qlib-DDB

微软 [QLib](https://github.com/microsoft/qlib) 量化投资平台的增强分支，核心特性是添加了 **DolphinDB** 高性能时序数据库后端支持，同时保持与原版 QLib 的完全兼容。

## 特性

- **DolphinDB 后端**: 通过 URI 自动切换后端，零代码修改
- **表达式计算**: 350+ 预置 Alpha 因子，在 DolphinDB 端高性能计算
- **Pyecharts 可视化**: 内置交互式图表（IC 分析、风险分析、日历热力图等）
- **因子分析**: 完整的因子统计、分组收益、IC 分析工具链
- **安全**: URI 凭据脱敏、安全的反序列化机制

## TODO

- [x] 接入 DolphinDB
- [x] 接入 DolphinDB 表达式计算
- [x] Pyecharts 交互式可视化
- [x] 因子统计分析模块
- [ ] 优化对分钟及 Tick 级别的支持

---

## 使用说明

### 1. 部署 DolphinDB

参看 DolphinDB 官方说明 [单节点部署与升级](https://docs.dolphindb.cn/zh/tutorials/standalone_server.html)

### 2. 安装依赖

```bash
pip install dolphindb
```

### 3. 创建 DolphinDB 库

```python
from qlib.data.backend.ddb_qlib import (
    create_feature_daily_table,
    create_calendar_table,
    create_instrument_table,
)

# 连接格式: dolphindb://用户名:密码@主机:端口
uri = "dolphindb://admin:123456@localhost:8848"

# 创建特征数据表
create_feature_daily_table(uri)

# 创建股票信息表
create_instrument_table(uri)

# 创建交易日历表
create_calendar_table(uri)
```

或使用脚本（默认连接，可用 `--ddb_uri` 参数指定）：

```bash
cd qlib-ddb/examples/初始创建ddb数据库
python test_ddb_op.py --clean_db True
```

### 4. 导入数据

从 WIND MySQL 数据库同步数据到 DolphinDB：

```python
from qlib.data.backend.ddb_qlib import init_qlib_ddb_from_mysql

ddb_uri = "dolphindb://admin:123456@localhost:8848"
mysql_uri = "mysql+mysqlconnector://root:123456@localhost:3306/windDB"

init_qlib_ddb_from_mysql(ddb_uri, mysql_uri)
```

或使用脚本：

```bash
cd qlib-ddb/examples/初始创建ddb数据库
python syn_mysql_to_ddb.py --ddb_uri "dolphindb://..." --mysql_uri "mysql+..."
```

### 5. 使用 Qlib

#### 5.1 DolphinDB 作为数据源

```python
import qlib
from qlib.constant import REG_CN
from qlib.data import D

# 初始化（使用 dolphindb:// 前缀）
uri = "dolphindb://admin:123456@localhost:8848"
qlib.init(database_uri=uri, region=REG_CN)

# 获取股票池
pool = D.instruments("ashares")
codes = D.list_instruments(pool, as_list=True)

# 获取交易日历
calendar = D.calendar()

# 使用 QLib 标准表达式
D.features(
    codes,
    start_time="2021-01-01",
    end_time="2024-04-12",
    fields=["Ref($S_DQ_ADJCLOSE, -2)/Ref($S_DQ_ADJCLOSE, -1) - 1"],
)
```

#### 5.2 使用 DolphinDB 表达式计算

系统在 `qlib/data/backend/ddb_qlib/ddb_scripts/` 中预置了 Alpha 因子表达式：

```python
D.features(
    instruments=codes,
    fields=[
        "WQAlpha1($close)",                              # WorldQuant 101 Alpha #1
        "gtjaAlpha1($open, $close, $volume)",            # 国泰君安 191 Alpha #1
        "qlib158Alpha1($open, $close)",                   # Alpha158 #1
        "$high",                                          # 基础字段
        "Ref($S_DQ_ADJCLOSE, -2)/Ref($S_DQ_ADJCLOSE, -1) - 1",  # 自定义表达式
    ],
    start_time="2020-01-01",
    end_time="2023-12-31",
    freq="day",
)
```

预置因子库：

| 因子库 | 范围 | 脚本文件 |
|--------|------|----------|
| WorldQuant 101 Alpha | `WQAlpha1` ~ `WQAlpha101` | `wq101alpha.dos` |
| 国泰君安 191 Alpha | `gtjaAlpha1` ~ `gtjaAlpha191` | `gtja191Alpha.dos` |
| Alpha158 | `qlib158Alpha1` ~ `qlib158Alpha158` | `qlib158Alpha.dos` |

#### 5.3 Alpha158 因子计算

```python
from qlib.contrib.data.loader import Alpha158DL
from qlib.data.dataset.loader import QlibDataLoader

conf = {
    "kbar": {},
    "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
    "rolling": {},
}

alpha158_expression, alpha158_names = Alpha158DL().get_feature_config(conf)

qdl = QlibDataLoader(config=(alpha158_expression, alpha158_names))
dataset = qdl.load(instruments=codes, start_time="2021-01-01", end_time="2023-12-31")
```

#### 5.4 DolphinDBDataLoader（直接表查询）

```python
from qlib.data.dataset.loader import DolphinDBDataLoader

uri = "dolphindb://admin:123456@localhost:8848"
qlib.init(database_uri=uri, region=REG_CN)

loader = DolphinDBDataLoader(
    table_name="stockDerivative",    # 表名（第一个参数）
    db_name="DailyBase",             # 数据库名（可选）
    config={
        "fields": ["S_PQ_HIGH_52W_", "S_VAL_PE"],
        "datetime_colName": "TRADE_DT",
        "instruments_colName": "S_INFO_WINDCODE",
        "pivot": False,
    },
)

data = loader.load(start_time="2025-01-01", end_time="2025-01-31")
```

---

## 可视化与分析

### Pyecharts 交互式图表

Qlib-DDB 内置了基于 Pyecharts 的可视化模块，提供交互式图表：

```python
from qlib.contrib.report.graph import (
    plot_bar,
    plot_timeseries,
    plot_distribution,
    plot_qq,
    CalendarGraph,     # 日历热力图
    ScatterGraph,      # 散点图
    BarGraph,          # 柱状图
    DistplotGraph,     # 分布图
)
```

### 模型性能分析

```python
from qlib.contrib.report.analysis_model.analysis_model_performance import (
    model_performance_graph,
    compute_group_return,
    compute_ic,
)

# 生成完整的模型性能报告（IC、分组收益、 autocorrelation 等）
model_performance_graph(pred_label, show_notebook=True)
```

### 风险分析

```python
from qlib.contrib.report.analysis_position.risk_analysis import risk_analysis_graph

# 生成风险分析图表（累计收益、回撤、波动率等）
risk_analysis_graph(analysis_df, report_normal_df, show_notebook=True)
```

### 因子统计分析

```python
from qlib.contrib.report.analysis_alpha.factor_stats import (
    FactorAnalyzer,
    run_factor_analysis,
)

# 一键运行完整因子分析（IC、分组收益、换手率等）
result = run_factor_analysis(
    factor_data=factor_df,
    return_data=return_df,
    group_count=5,
)
```

### 图表显示配置

```python
from qlib.contrib.report.display_config import ReportGraphConfig

# 自定义图表配置（图例、标题、颜色等）
config = ReportGraphConfig(
    width="100%",
    height=500,
    legend_position="top",
)
```

---

## 安全特性

### URI 凭据脱敏

日志中自动隐藏密码，防止敏感信息泄露：

```python
# 实际连接
dolphindb://admin:123456@172.17.0.1:8848

# 日志输出
DolphinDB(dolphindb://admin:***@172.17.0.1:8848)
```

默认启用脱敏。如需禁用（不推荐）：

```python
qlib.init(database_uri=uri, log_mask_sensitive=False)
```

---

## 项目结构

```
qlib/
├── data/
│   ├── backend/ddb_qlib/          # DolphinDB 后端
│   │   ├── ddb_client.py          # 连接和会话管理
│   │   ├── ddb_operator.py        # 核心数据库操作
│   │   ├── ddb_mysql_bridge.py    # MySQL → DolphinDB 数据同步
│   │   ├── ddb_features.py        # 表达式系统（73 个操作符映射）
│   │   ├── schemas.py             # 表结构定义
│   │   ├── utils.py               # 工具函数
│   │   └── ddb_scripts/           # Alpha 因子表达式库
│   │       ├── wq101alpha.dos     # WorldQuant 101 Alpha
│   │       ├── gtja191Alpha.dos   # 国泰君安 191 Alpha
│   │       └── qlib158Alpha.dos   # Alpha158 因子
│   └── storage/
│       └── dolphindb_storage.py   # DolphinDB Storage 层实现
├── contrib/
│   ├── report/
│   │   ├── graph.py               # Pyecharts 图表工具
│   │   ├── display_config.py      # 图表显示配置
│   │   ├── analysis_alpha/
│   │   │   └── factor_stats.py    # 因子统计分析
│   │   ├── analysis_model/
│   │   │   └── analysis_model_performance.py  # 模型性能分析
│   │   └── analysis_position/
│   │       ├── risk_analysis.py   # 风险分析
│   │       └── score_ic.py        # Score IC 分析
│   └── strategy/
│       └── signal_strategy.py     # 信号策略
└── utils/
    └── pickle_utils.py            # 安全的反序列化工具
```

---

## 更多 Qlib 模型

可以加入星球获取更多内容：

![image](https://github.com/hugo2046/qlib-ddb/raw/2230f4533b83268acdd29e4e2e472115a5083aa9/imgs/%E7%9F%A5%E8%AF%86%E6%98%9F%E7%90%83.jpg)
