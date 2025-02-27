<!--
 * @Author: hugo2046 shen.lan123@gmail.com
 * @Date: 2025-02-18 11:26:04
 * @LastEditors: hugo2046 shen.lan123@gmail.com
 * @LastEditTime: 2025-02-26 17:20:38
 * @FilePath: /workspace/qlib-ddb/README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
[![Python Versions](https://img.shields.io/pypi/pyversions/pyqlib.svg?logo=python&logoColor=white)](https://pypi.org/project/pyqlib/#files)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos-lightgrey)](https://pypi.org/project/pyqlib/#files)
[![PypI Versions](https://img.shields.io/pypi/v/pyqlib)](https://pypi.org/project/pyqlib/#history)
[![Upload Python Package](https://github.com/microsoft/qlib/workflows/Upload%20Python%20Package/badge.svg)](https://pypi.org/project/pyqlib/)
[![Github Actions Test Status](https://github.com/microsoft/qlib/workflows/Test/badge.svg?branch=main)](https://github.com/microsoft/qlib/actions)
[![Documentation Status](https://readthedocs.org/projects/qlib/badge/?version=latest)](https://qlib.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/pypi/l/pyqlib)](LICENSE)
[![Join the chat at https://gitter.im/Microsoft/qlib](https://badges.gitter.im/Microsoft/qlib.svg)](https://gitter.im/Microsoft/qlib?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

## :newspaper: **What's NEW!** &nbsp;   :sparkling_heart: 

Recent released features

## TODO
- [x] 接入DolphinDB
- [ ] 优化对分钟及Tick级别的支持
- [ ] 接入DolphinDB表达式计算


## 使说明

### 1.首先部署DolphindD
参看DolphinDB官方说明[单节点部署与升级](https://docs.dolphindb.cn/zh/tutorials/standalone_server.html)

### 2.依赖
同时依赖python下的dolphindb包
```bash
pip install dolphindb
```

### 3.建立DolphinDB库用于Qlib连接
此部分代码在qlib/data/backend/ddb_qlib下,详见:[qlib-ddb](https://github.com/hugo2046/qlib-ddb/tree/main/qlib/data/backend/ddb_qlib)

**创建DolphinDB库**
```python

from qlib.data.backend.ddb_qlib import create_qlib_table
from qlib.data.backend.ddb_qlib.schemas import QlibTableSchema

# 连接格式
uri = "dolphin://admin:123456@localhost:8848"

# 创建特征数据表
create_feature_daily_table(uri)

# 创建股票信息表
create_instrument_table(uri)

# 创建交易日历表
create_calendar_table(uri)
```
或者直接使用脚本,这样直接使用ddb默认连接创建，也可以使用`--ddb_uri`参数指定

```bash
cd /qlib-ddb/examples/初始创建ddb数据库
python test_ddb_op.py --clean_db True
```


**导入数据**
如果是WIND数据库可以直接从mysql同步过来
```python

from .ddb_operator import cinit_qlib_ddb_from_mysql

ddb_uri: str = "dolphin://admin:123456@localhost:8848"
mysql_uri: str = "mysql+mysqlconnector://root:123456@localhost:3306/windDB"

# 测试导入初始数据
init_qlib_ddb_from_mysql(ddb_uri, mysql_uri)
```
或者使用脚本从WIND数据库同步数据到DolphinDB数据库,如果没参数则使用默认连接,这里有两个参数`--ddb_uri`和`--mysql_uri`,可以自行指定。
```bash
cd /qlib-ddb/examples/初始创建ddb数据库
python syn_mysql_to_ddb.py 
```


### 4.使用Qlib

```python
import qlib
from qlib.constant import REG_CN
from qlib.data import D
import pandas as pd


# 初始化qlib
uri = "dolphin://admin:123456@localhost:8848"
qlib.init(database_uri=uri,region=REG_CN)


# 获取股票池
pool = D.instruments("ashares")
codes:List[str] = D.list_instruments(pool, as_list=True)
print("股票池:")
print(codes[:10])

# 获取日历
calcendar = D.calendar()
print("交易日历:")
print(calcendar[-10:])

# 根据表达式计算
D.features(
    codes,
    start_time="2021-01-01",
    end_time="2024-04-12",
    fields=["Ref($S_DQ_ADJCLOSE, -2)/Ref($S_DQ_ADJCLOSE, -1) - 1"],
)

```