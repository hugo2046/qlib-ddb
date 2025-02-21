# DDB QLib 数据连接工具

## 项目概述

本工具提供DolphinDB与MySQL数据库间的数据同步能力，支持QLib金融研究框架的数据初始化需求。

## 主要功能

✅ 双向数据同步
✅ 自动类型转换
✅ 批量写入优化
✅ 异常处理机制

## 快速开始

### 依赖安装
```bash
pip install dolphindb mysqlclient pandas
```

### 基础使用
```python
from ddb_qlib import DDBMySQLBridge, init_qlib_ddb_from_mysql

# 初始化连接
bridge = DDBMySQLBridge(
    ddb_uri='dolphindb://user:pass@host:port',
    mysql_uri='mysql://user:pass@host:port/db'
)

# 同步全量数据
init_qlib_ddb_from_mysql(bridge.ddb_uri, bridge.mysql_uri)
```

## 配置说明

在`config.py`中配置数据库连接参数：
```python
# DolphinDB 连接配置
DDB_CONFIG = {
    'host': 'your_host',
    'port': 8848,
    'user': 'admin',
    'password': '123456'
}

# MySQL 连接配置
MYSQL_CONFIG = {
    'host': 'mysql_host',
    'database': 'windDB',
    'user': 'devuser',
    'password': 'Knight@678'
}
```

## 注意事项

1. 确保DolphinDB服务已安装MySQL插件
2. 生产环境建议使用连接池
3. 敏感信息请通过环境变量配置
4. 大数据量操作建议分批次执行

## 技术支持

遇到问题请联系：shen.lan123@gmail.com
