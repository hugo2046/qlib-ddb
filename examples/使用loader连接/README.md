```python
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = str(Path().cwd().parents[1].resolve())
sys.path.insert(0, PROJECT_ROOT)

import qlib
from qlib.constant import REG_CN
from qlib.data.dataset.loader import DolphinDBDataLoader, MySQLDataLoader
```

# 说明

`DolphinDBDataLoader`, `MySQLDataLoader`分别用于连接DolphinDB和MySQL数据库,类似于`StaticDataLoader`模块的做用,可以直接从数据库中加载数据,而不需要事先将数据下载到本地。

用以支持简单数据或储存好的因子数据传入qlib模型训练和预测，传入过程类似:

```mermaid
graph TB
	StaticDataLoader-->DataHandlerLP;
	DolphinDBDataLoader-->DataHandlerLP;
	MySQLDataLoader-->DataHandlerLP;
	DataHandlerLP-->DatasetH;
	DatasetH-->Model[模型训练];
	Model-->analyizer[分析回测];
```

## MySQLLoader

参数`db_name`需要查询的数据库名,`table_name`需要查询的表名,`connect`为连接数据库的url,格式如下:
```python

url: str = "mysql+mysql://用户名称:密码@IP地址:端口号"
msql_loader = MySQLDataLoader(
    db_name="factor_db",
    table_name="daily_factor",
    connect=url,
    config={
        "fields": [
            "forecast_est_oper_cost_rolling_180d_cagr",
            "forecast_est_oper_profit_rolling_180d_cagr",
        ],
        "datetime_colName": "trade_date",
        "instruments_colName": "code",
        "columns": "factor_name",
        "values": "factor_value",
        "pivot": True,
    },
)
```
`config`参数注意是字典格式。`pivot`为True是,会将窄表转为宽表。根据`datetime_colName`将对应字段转为index,`instruments_colName`将对应字段转为columns,需要指定`columns`和`values`。由于是窄表所有会根据指定的columns查询对应fields并将values作为值。

窄表结构类似于：
|trade_date|code|factor_name|factor_value|
|---|---|---|---|
|2023-01-03|000001.SZ|forecast_est_oper_cost_rolling_180d_cagr|0.123|
|2023-01-03|000001.SZ|forecast_est_oper_profit_rolling_180d_cagr|0.456|
|2023-01-03|000002.SZ|forecast_est_oper_cost_rolling_180d_cagr|0.789|
|2023-01-03|000002.SZ|forecast_est_oper_profit_rolling_180d_cagr|0.012|

上述表格中,`trade_date`为时间列,`code`为标的列,`factor_name`为因子名称,`factor_value`为因子值。会根据`config`参数将表格转为宽表,类似于:

|trade_date|code|forecast_est_oper_cost_rolling_180d_cagr|forecast_est_oper_profit_rolling_180d_cagr|
|---|---|---|---|
|2023-01-03|000001.SZ|0.123|0.456|
|2023-01-03|000002.SZ|0.789|0.012|

上述config的过程类似先使用mysql语句类似于:

- step 1: 查询数据
    ```sql
    SELECT trade_date, code, factor_name, factor_value FROM factor_db.daily_factor 
    WHERE factor_name IN ('forecast_est_oper_cost_rolling_180d_cagr', 'forecast_est_oper_profit_rolling_180d_cagr');
    ```
- step 2: 转换数据
    查询后在使用pd.pivot_table()进行转换。
    ```python
    pd.pivot_table(data, index=['trade_date', 'code'], columns='factor_name', values='factor_value')
    ```

- step3:数据获取
    `load`方法于StaticDataLoader类似,可以指定时间区间,返回对应的DataFrame数据。如果`instruments`参数不指定,则返回所有标的。
    ```python
    example_df1: pd.DataFrame = msql_loader.load(
    start_time="2025-01-01",
    end_time="2025-01-31",
    )
    ```

**注意**:
1. 使用`MySQLDataLoader`需要安装`pymysql`,`SQLalchemy`包。
2. 使用MySQLDataLoader需要保证数据库可以远程连接,如果是本地数据库,则`connect`参数可以设置为`mysql+pymysql://用户名称:密码@localhost:端口号`。
3. `datetime_format`参数可以指定时间格式化,默认格式为`%Y-%m-%d`,如果数据库中时间格式为`20250101`,则需要指定`datetime_format="%Y%m%d"`。
4. 如果数据表为宽表,则不需要指定`columns`和`values`,并且`pivot`参数设置为False即可。
5. 如果数据表为宽表,则`fields`参数可以不指定,默认查询所有字段。
6. 无需使用qlib.init()进行初始化,直接使用即可。

### 窄表查询


```python
url: str = "mysql+pymysql://username:password@ip:prot"
msql_loader = MySQLDataLoader(
    db_name="stocklink",
    table_name="daily_factor",
    connect=url,
    config={
        "fields": [
            "forecast_est_oper_cost_rolling_180d_cagr",
            "forecast_est_oper_profit_rolling_180d_cagr",
        ],
        "datetime_colName": "trade_date",
        "instruments_colName": "code",
        "columns": "factor_name",
        "values": "factor_value",
        "pivot": True,  # 是否需要透视表，针对窄表设计
    },
)
```


```python
# load方法参数
example_df1: pd.DataFrame = msql_loader.load(
    start_time="2025-01-01",
    end_time="2025-01-31",
)
```


```python
example_df1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>factor_name</th>
      <th>forecast_est_oper_cost_rolling_180d_cagr</th>
      <th>forecast_est_oper_profit_rolling_180d_cagr</th>
    </tr>
    <tr>
      <th>trade_date</th>
      <th>code</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">2025-01-02</th>
      <th>000001.SZ</th>
      <td>NaN</td>
      <td>0.084630</td>
    </tr>
    <tr>
      <th>000002.SZ</th>
      <td>-0.153877</td>
      <td>-0.822999</td>
    </tr>
    <tr>
      <th>000008.SZ</th>
      <td>-0.030454</td>
      <td>-0.813660</td>
    </tr>
    <tr>
      <th>000012.SZ</th>
      <td>-0.003310</td>
      <td>-0.177573</td>
    </tr>
    <tr>
      <th>000026.SZ</th>
      <td>-0.024027</td>
      <td>-0.090818</td>
    </tr>
  </tbody>
</table>
</div>




```python
# load方法参数
example_df2: pd.DataFrame = msql_loader.load(
    instruments=["000001.SZ", "000002.SZ"],
    start_time="2025-01-01",
    end_time="2025-01-31",
)
```


```python
example_df2.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>factor_name</th>
      <th>forecast_est_oper_cost_rolling_180d_cagr</th>
      <th>forecast_est_oper_profit_rolling_180d_cagr</th>
    </tr>
    <tr>
      <th>trade_date</th>
      <th>code</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2025-01-23</th>
      <th>000002.SZ</th>
      <td>-0.147549</td>
      <td>-0.915713</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2025-01-24</th>
      <th>000001.SZ</th>
      <td>NaN</td>
      <td>0.084630</td>
    </tr>
    <tr>
      <th>000002.SZ</th>
      <td>-0.147549</td>
      <td>-0.915713</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2025-01-27</th>
      <th>000001.SZ</th>
      <td>NaN</td>
      <td>0.084630</td>
    </tr>
    <tr>
      <th>000002.SZ</th>
      <td>-0.151105</td>
      <td>-0.915713</td>
    </tr>
  </tbody>
</table>
</div>




```python
example_df2.index.levels[1]
```




    Index(['000001.SZ', '000002.SZ'], dtype='object', name='code')



### 宽表查询


```python
url: str = "mysql+pymysql://username:password@ip:port"
msql_loader2 = MySQLDataLoader(
    db_name="windDB",
    table_name="ASHAREEODPRICES",
    connect=url,
    config={
        "fields": [
            "S_DQ_ADJCLOSE",
            "S_DQ_VOLUME",
        ],
        "datetime_colName": "TRADE_DT",
        "instruments_colName": "S_INFO_WINDCODE",
        "datetime_format": "%Y%m%d",  # 日期格式化
        "pivot": False,  # 是否需要透视表，针对窄表设计
    },
)
```


```python
example_df3: pd.DataFrame = msql_loader2.load(
    start_time="20250101",
    end_time="20250131",
)
```


```python
example_df3.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>S_DQ_ADJCLOSE</th>
      <th>S_DQ_VOLUME</th>
    </tr>
    <tr>
      <th>TRADE_DT</th>
      <th>S_INFO_WINDCODE</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">2025-01-02</th>
      <th>832876.BJ</th>
      <td>44.3811</td>
      <td>11921.21</td>
    </tr>
    <tr>
      <th>002917.SZ</th>
      <td>23.3019</td>
      <td>91972.71</td>
    </tr>
    <tr>
      <th>002256.SZ</th>
      <td>36.9990</td>
      <td>463626.00</td>
    </tr>
    <tr>
      <th>300331.SZ</th>
      <td>64.7908</td>
      <td>66369.66</td>
    </tr>
    <tr>
      <th>601156.SH</th>
      <td>17.8867</td>
      <td>137364.09</td>
    </tr>
  </tbody>
</table>
</div>



## DolphinDBDataLoader

通过qlib.init初始化后,可以直接使用DolphinDBDataLoader加载数据,类似于MySQLDataloader的用法,只需要指定数据库名和表名即可。


```python
uri = "dolphindb://username:password@ip:port"
qlib.init(database_uri=uri, region=REG_CN)
```

    [3945290:MainThread](2025-09-24 11:47:06,878) INFO - qlib.Initialization - [config.py:438] - default_conf: client.
    [3945290:MainThread](2025-09-24 11:47:07,452) INFO - qlib.ddb_features - [ddb_features.py:90] - 已注册 qlib 兼容函数到 DolphinDB 会话
    [3945290:MainThread](2025-09-24 11:47:07,454) INFO - qlib.Initialization - [__init__.py:86] - qlib successfully initialized based on client settings.
    [3945290:MainThread](2025-09-24 11:47:07,455) INFO - qlib.Initialization - [__init__.py:96] - data_path={'__DEFAULT_FREQ': 'DolphinDB(dolphindb://admin:123456@locahost:8848)'}



```python
# 通过init初始化后，可以直接使用DolphinDBDataLoader加载数据
ddb_Loader = DolphinDBDataLoader(
    db_name="DailyBase",
    table_name="stockDerivative",
    config={
        "fields": ["S_PQ_HIGH_52W_", "S_VAL_PE"],
        "datetime_colName": "TRADE_DT",
        "instruments_colName": "S_INFO_WINDCODE",
        "pivot": False,
    },
)
```


```python
ddb_Loader.load(
    start_time="2025-01-01",
    end_time="2025-01-31",
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>S_PQ_HIGH_52W_</th>
      <th>S_VAL_PE</th>
    </tr>
    <tr>
      <th>TRADE_DT</th>
      <th>S_INFO_WINDCODE</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">2025-01-02</th>
      <th>600230.SH</th>
      <td>13.89</td>
      <td>22.6319</td>
    </tr>
    <tr>
      <th>301552.SZ</th>
      <td>83.08</td>
      <td>28.1303</td>
    </tr>
    <tr>
      <th>688528.SH</th>
      <td>13.99</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>605028.SH</th>
      <td>26.00</td>
      <td>14.1318</td>
    </tr>
    <tr>
      <th>600237.SH</th>
      <td>8.57</td>
      <td>46.6784</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">2025-01-31</th>
      <th>002452.SZ</th>
      <td>9.27</td>
      <td>27.0603</td>
    </tr>
    <tr>
      <th>688088.SH</th>
      <td>51.84</td>
      <td>223.5079</td>
    </tr>
    <tr>
      <th>002853.SZ</th>
      <td>13.43</td>
      <td>24.6036</td>
    </tr>
    <tr>
      <th>002044.SZ</th>
      <td>5.72</td>
      <td>33.7528</td>
    </tr>
    <tr>
      <th>000151.SZ</th>
      <td>18.90</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>118581 rows × 2 columns</p>
</div>


