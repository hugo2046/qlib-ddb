"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2025-02-20 15:41:01
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2025-02-20 15:42:54
Description: 用于qlib的表及数据库结构

由于ddb同一个数据库中储存的必须是相同结构的表，所以这里会创建三个库，每个库中会有一个表

如果需要使用qlib的因子则需要使用默认的数据字段，否则可以使用自己的数据字段
qlib默认数据为后复权数据，复权因子为后复权因子。在回测时使用后复权因子还原价格。
date,open,high,low,close,amount,volume,vwap,factor
"""

from pydantic import BaseModel, field_validator
from typing import List, Tuple, Optional,Dict,Union
import dolphindb as ddb
import pandas as pd
import numpy as np
from packaging import version

FIELDS_MAPPING: Dict = {
    "TRADE_DT": "date",
    "S_INFO_WINDCODE": "code",
    "S_DQ_ADJOPEN": "open", # 后复权开盘价
    "S_DQ_ADJHIGH": "high",# 后复权最高价
    "S_DQ_ADJLOW": "low", # 后复权最低价
    "S_DQ_ADJCLOSE": "close", # 后复权收盘价
    "S_DQ_VOLUME": "volume", # 成交量(手)
    "S_DQ_AMOUNT": "amount", # 成交金额(千元)
    "S_DQ_ADJFACTOR": "factor", # 后复权因子
    "S_DQ_AVGPRICE": "vwap", # 均价
}

def get_year_end_freq():
    """
    根据 pandas 版本获取年末 ('Year-End') 的频率字符串。
    - pandas < 2.0.0 使用 'A'
    - pandas >= 2.0.0 使用 'YE'
    """
    if version.parse(pd.__version__) < version.parse("2.0.0"):
        return 'A'
    else:
        return 'YE'



class TableSchema(BaseModel):
    db_name: str
    table_name: str
    columns: List[Tuple[str, str]]  # (列名, 类型)
    partition_type: int  # ddb.settings.* ,为int类型
    partition_columns: Optional[Union[str, List[str]]] = None,
    engine: str = "OLAP"
    primary_key: Optional[Union[str, List[str]]] = None
    partitions: Optional[List] = None

    @field_validator("partition_type")
    @classmethod
    def validate_partition_type(cls, value: int) -> int:
        valid_types = {
            ddb.settings.VALUE,
            ddb.settings.RANGE,
            ddb.settings.LIST,
            ddb.settings.HASH,
            ddb.settings.COMPO,
        }
        if value not in valid_types:
            raise ValueError(f"无效的分区类型: {value}，有效值: {valid_types}")
        return value

    def map_columns_to_fields(self) -> List[str]:   
        mapping:Dict = {v:k for k,v in FIELDS_MAPPING.items()}
        return [mapping.get(col, col) for col, _ in self.columns]


class QlibTableSchema:
    """QLib标准表配置"""

    @classmethod
    def feature_daily(cls) -> TableSchema:
        # storage_name,instrument,freq
        freq = get_year_end_freq()
        return TableSchema(
            db_name="QlibFeaturesDay",
            table_name="Features",
            columns=[
                ("date", "DATE"),
                ("code", "SYMBOL"),
                ("open", "DOUBLE"),
                ("high", "DOUBLE"),
                ("low", "DOUBLE"),
                ("close", "DOUBLE"),
                ("volume", "DOUBLE"),
                ("amount", "DOUBLE"),
                ("factor", "DOUBLE"),
                ("vwap", "DOUBLE"),
            ],
            partition_type=ddb.settings.RANGE,
            partition_columns="date",
            partitions=np.array(
                pd.date_range("2010-01-01", "2045-12-31", freq=freq),
                dtype="datetime64[M]",
            ),
        )

    @classmethod
    def instrument(cls,table_name:str="ashares") -> TableSchema:
        # qlib的调用为instrument.xxxx
        return TableSchema(
            db_name="QlibInstruments",
            table_name=table_name,
            columns=[
                ("S_INFO_WINDCODE", "SYMBOL"),
                ("S_INFO_LISTDATE", "DATE"),
                ("S_INFO_DELISTDATE", "DATE"),
            ],
            partition_type=ddb.settings.HASH,
            partition_columns="S_INFO_WINDCODE",
            engine="PKEY",
            primary_key=["S_INFO_WINDCODE","S_INFO_LISTDATE","S_INFO_DELISTDATE"],
            partitions=[ddb.settings.DT_SYMBOL, 20],
        )

    @classmethod
    def calendar(cls) -> TableSchema:
        # FIXME:
        # 当freq不为day时，qlib会报错
        # qlib的调用为calendar.xxxx 默认freq为day
        freq = get_year_end_freq()
        return TableSchema(
            db_name="QlibCalendars",
            table_name="day",
            columns=[("TRADE_DAYS", "DATE")],
            partition_type=ddb.settings.RANGE,
            partition_columns="TRADE_DAYS",
            engine="PKEY",
            primary_key="TRADE_DAYS",
            partitions=np.array(
                pd.date_range("2000-01-01", "2060-12-31", freq=freq),
                dtype="datetime64[M]",
            ),
        )

    @classmethod
    def get_all_databases(cls) -> list:
        """获取所有QLib数据库名称"""
        return list(
            {
                cls.feature_daily().db_name,
                cls.instrument().db_name,
                cls.calendar().db_name,
            }
        )
