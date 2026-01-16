<!--
 * @Author: hugo2046 shen.lan123@gmail.com
 * @Date: 2025-02-18 11:26:04
 * @LastEditors: shen.lan123@gmail.com
 * @LastEditTime: 2025-07-23 21:36:53
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
- [x] 接入DolphinDB表达式计算


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

#### 4.1 使用DolphinDB作为数据源

使用`D.features`获取数据或进行表达式计算

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

使用`QlibDataLoader`计算alpha158因子

```python
from qlib.contrib.data.loader import Alpha158DL
from qlib.data.dataset.loader import QlibDataLoader

conf = {
    "kbar": {},
    "price": {
        "windows": [0],
        "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
    },
    "rolling": {},
}

alpha158:List[List] = Alpha158DL().get_feature_config(conf)

# 获取因子表达
alpha158_expression = alpha158[0]
# 获取因子名
alpha158_names = alpha158[1]

# 因子计算
qdl:QlibDataLoader = QlibDataLoader(config=(alpha158_expression, alpha158_names))

dataset:pd.DataFrame = qdl.load(instruments=codes, start_time="2021-01-01", end_time="2023-12-31")
```

#### 4.2 使用DolphinDB表达式计算

Qlib-DDB 支持直接使用 DolphinDB 表达式进行因子计算。系统已在 `qlib.data.backend.ddb_qlib.ddb_scripts` 中预置了 **WorldQuant 101 Alpha 因子**、**国泰君安 191 Alpha 因子**和**Alpha158因子**的表达式计算脚本。

使用 `D.features` 进行 DolphinDB 表达式计算：

```python
# 使用预置的 Alpha 因子表达式
D.features(
    instruments=codes,
    fields=[
        "WQAlpha1($close)",           # WorldQuant 101 Alpha 因子 #1
        "gtjaAlpha1($open, $close, $volume)",  # 国泰君安 191 Alpha 因子 #1
        "qlib158Alpha1($open, $close)",        # Alpha158 因子 #1
        "$high",                               # 基础字段
        "Ref($S_DQ_ADJCLOSE, -2)/Ref($S_DQ_ADJCLOSE, -1) - 1",  # 自定义表达式
    ],
    start_time="2020-01-01",
    end_time="2023-12-31",
    freq="day",
)
```

##### 预置因子说明：

- **WorldQuant Alpha 因子**: `WQAlpha1` ~ `WQAlpha101` - 101个经典Alpha因子
- **国泰君安 Alpha 因子**: `gtjaAlpha1` ~ `gtjaAlpha191` - 191个量化因子
- **Alpha158 因子**: `qlib158Alpha1` ~ `qlib158Alpha158` - 158个机器学习因子

#### 4.3 使用DolphinDBDataLoader

对于更灵活的数据加载需求，可以使用 `DolphinDBDataLoader` 直接从 DolphinDB 数据库加载数据：

```python
from qlib.data.dataset.loader import DolphinDBDataLoader

# 初始化 Qlib
uri = "dolphindb://admin:123456@localhost:8848"
qlib.init(database_uri=uri, region=REG_CN)

# 创建 DolphinDB 数据加载器
loader = DolphinDBDataLoader(
    db_name="DailyBase",           # 数据库名
    table_name="stockDerivative",  # 表名
    config={
        "fields": ["S_PQ_HIGH_52W_", "S_VAL_PE"],  # 要查询的字段
        "datetime_colName": "TRADE_DT",              # 时间列名
        "instruments_colName": "S_INFO_WINDCODE",     # 股票代码列名
        "pivot": False,                               # 是否进行透视变换
    },
)

# 加载数据
data = loader.load(
    start_time="2025-01-01",
    end_time="2025-01-31",
)
print(data.head())
```
## 更多qlib模型可以加入星球
![image](https://github.com/hugo2046/qlib-ddb/raw/2230f4533b83268acdd29e4e2e472115a5083aa9/imgs/%E7%9F%A5%E8%AF%86%E6%98%9F%E7%90%83.jpg)

