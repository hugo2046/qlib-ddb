# DDB QLib - DolphinDB 后端存储适配器

## 项目概述

`ddb_qlib` 是一个为 QLib 量化投资平台提供 DolphinDB 后端存储支持的适配器。它允许用户使用高性能的 DolphinDB 时序数据库作为 QLib 的数据源，实现金融数据的高效存储和快速访问。

## 核心特性

### 1. 灵活的数据库连接
✅ 标准化的 URI 连接方式
```python
uri = "dolphindb://admin:123456@host:port"
```
✅ 支持连接参数验证和安全检查
✅ 自动重连和会话管理

### 2. 专业的金融数据结构
✅ 预置 QLib 标准表结构：
- `Features` 表：日频因子数据
- `Instruments` 表：股票基础信息
- `Calendar` 表：交易日历

### 3. 优化的存储设计
✅ 分区存储策略：
- 特征数据：按时间范围分区
- 股票数据：按股票代码哈希分区
- 日历数据：按时间范围分区

### 4. 高效的数据操作
✅ 批量数据处理
✅ 自动类型转换
✅ 异常处理机制

## 快速开始

### 1. 安装依赖
```bash
pip install dolphindb pandas numpy pydantic
```

### 2. 初始化 QLib
```python
import qlib
from qlib.constant import REG_CN

# 使用 DolphinDB 作为数据源
qlib.init(
    database_uri="dolphindb://admin:123456@host:port",
    region=REG_CN
)
```

### 3. 创建数据表
```python
from qlib.data.backend.ddb_qlib import create_qlib_table
from qlib.data.backend.ddb_qlib.schemas import QlibTableSchema

# 创建特征数据表
create_feature_daily_table(uri)

# 创建股票信息表
create_instrument_table(uri)

# 创建交易日历表
create_calendar_table(uri)
```

### 4. MySQL 数据同步

#### 4.1 基本用法
```python
from qlib.data.backend.ddb_qlib import DDBMySQLBridge, init_qlib_ddb_from_mysql

# 方法 1: 使用快速初始化函数
ddb_uri = "dolphindb://admin:123456@host:port"
mysql_uri = "mysql://user:pass@host:port/winddb"
init_qlib_ddb_from_mysql(ddb_uri, mysql_uri)

# 方法 2: 使用 Bridge 类进行灵活控制
bridge = DDBMySQLBridge(ddb_uri, mysql_uri)

# 检查 MySQL 表
all_tables = bridge.show_tables()
print("MySQL tables:", all_tables)

# 获取表结构
schema = bridge.extract_table_schema("wind_stock_daily")
print("Table schema:", schema)

# 加载数据
df = bridge.load_table(
    "wind_stock_daily",     # MySQL 表名
    table_schema={          # 类型映射
        "TRADE_DT": "DATE",
        "SYMBOL": "SYMBOL",
        "CLOSE": "DOUBLE"
    },
    start_row=0,           # 起始行
    row_num=10000         # 读取行数
)
```

#### 4.2 高级用法
```python
# 1. 安装 MySQL 插件
bridge.load_mysql_plugin()

# 2. 使用 SQL 查询加载数据
df = bridge.load_table(
    """SELECT * FROM wind_stock_daily 
       WHERE TRADE_DT >= '2010-01-01' 
       ORDER BY TRADE_DT ASC
       LIMIT 10000""",
    table_schema={                     # 类型映射
        "TRADE_DT": "DATE",
        "SYMBOL": "SYMBOL",
        "CLOSE": "DOUBLE"
    },
    allow_empty_table=True            # 允许空表
)

# 3. 关闭连接
bridge.close()
```

## 技术细节

### 表结构设计

1. **特征数据表 (Features)**
   - 日期、股票代码、开盘价、最高价、最低价、收盘价等
   - 按年份范围分区，优化时间序列查询

2. **股票信息表 (Instruments)**
   - 股票代码、上市日期、退市日期
   - 哈希分区，支持快速查找

3. **交易日历表 (Calendar)**
   - 交易日期
   - 范围分区，支持日期范围查询

### 性能优化

1. **分区策略**
   - 时间序列数据：RANGE 分区
   - 代码类数据：HASH 分区
   - 支持复合分区

2. **存储引擎**
   - OLAP：大规模数据分析
   - PKEY：高效的主键索引

### MySQL 集成机制

1. **连接管理**
   - 支持标准 MySQL URI 格式
   - 支持多种 MySQL 方言（mysql+pymysql，mysql+mysqldb）
   - 支持 SSL 加密连接

2. **数据加载**
   - 支持表名和 SQL 查询两种方式
   - 支持批量读取和分页加载
   - 支持灵活的类型映射配置

3. **异常处理**
   - 连接参数验证
   - 详细的错误日志
   - 支持空表处理

## 最佳实践

1. **数据库设计**
   - 根据数据特点选择合适的分区策略
   - 合理设置分区大小，避免过度分区
   - 确保 MySQL 和 DolphinDB 的表结构匹配

2. **性能优化**
   - 使用 SQL 查询进行数据过滤
   - 合理设置批量大小
   - 避免加载过大的数据集

3. **数据加载**
   - 先使用 extract_table_schema 获取表结构
   - 注意数据类型映射的准确性
   - 使用分页加载大表

4. **运维建议**
   - 及时关闭不用的连接
   - 做好异常处理和错误日志
   - 定期检查数据一致性

## 技术支持

如有问题，请联系：
- Email: shen.lan123@gmail.com
- GitHub Issues

## 许可证

本项目采用 MIT 许可证
