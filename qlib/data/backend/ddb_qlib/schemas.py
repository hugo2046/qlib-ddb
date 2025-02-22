"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2025-02-20 15:41:01
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2025-02-20 15:42:54
Description: 用于qlib的表及数据库结构

由于ddb同一个数据库中储存的必须是相同结构的表，所以这里会创建三个库，每个库中会有一个表
"""

from pydantic import BaseModel,field_validator
from typing import List, Tuple,Optional
import dolphindb as ddb
import pandas as pd
import numpy as np

class TableSchema(BaseModel):
    db_name: str
    table_name: str
    columns: List[Tuple[str, str]] # (列名, 类型)
    partition_type: int  # ddb.settings.* ,为int类型
    partition_columns: str
    engine: str = "OLAP"
    primary_key: Optional[str] = None
    partitions: Optional[List] = None

    @field_validator('partition_type')
    @classmethod
    def validate_partition_type(cls, value: int) -> int:
        valid_types = {
            ddb.settings.VALUE,
            ddb.settings.RANGE, 
            ddb.settings.LIST,
            ddb.settings.HASH,
            ddb.settings.COMPO
        }
        if value not in valid_types:
            raise ValueError(f"无效的分区类型: {value}，有效值: {valid_types}")
        return value


class QlibTableSchema:
    """QLib标准表配置"""

    @classmethod
    def feature_daily(cls) -> TableSchema:
        # storage_name,instrument,freq
        return TableSchema(
            db_name="QlibFeaturesDay",
            table_name="Features",
            columns=[
                ("TRADE_DT", "DATE"),
                ("S_INFO_WINDCODE", "SYMBOL"),
                ("S_DQ_ADJOPEN", "DOUBLE"),
                ("S_DQ_ADJHIGH", "DOUBLE"),
                ("S_DQ_ADJLOW", "DOUBLE"),
                ("S_DQ_ADJCLOSE", "DOUBLE"),
                ("S_DQ_VOLUME", "DOUBLE"),
                ("S_DQ_AMOUNT", "DOUBLE"),
                ("S_DQ_ADJFACTOR", "DOUBLE"),
                ("S_DQ_ADJPRECLOSE", "DOUBLE"),
            ],
            partition_type=ddb.settings.RANGE,
            partition_columns="TRADE_DT",
            partitions=np.array(
                pd.date_range("2010-01-01", "2045-12-31", freq="YE"),
                dtype="datetime64[M]",
            ),
        )

    @classmethod
    def instrument(cls) -> TableSchema:
        # qlib的调用为instrument.xxxx
        return TableSchema(
            db_name="QlibInstruments",
            table_name="ashares",
            columns=[
                ("S_INFO_WINDCODE", "SYMBOL"),
                ("S_INFO_LISTDATE", "DATE"),
                ("S_INFO_DELISTDATE", "DATE"),
            ],
            partition_type=ddb.settings.HASH,
            partition_columns="S_INFO_WINDCODE",
            engine="PKEY",
            primary_key="S_INFO_WINDCODE",
            partitions=[ddb.settings.DT_SYMBOL, 20],
        )

    @classmethod
    def calendar(cls) -> TableSchema:
        # qlib的调用为calendar.xxxx 默认freq为day
        return TableSchema(
            db_name="QlibCalendars",
            table_name="day",
            columns=[("TRADE_DAYS", "DATE")],
            partition_type=ddb.settings.RANGE,
            partition_columns="TRADE_DAYS",
            engine="PKEY",
            primary_key="TRADE_DAYS",
            partitions=np.array(
                pd.date_range("2010-01-01", "2045-12-31", freq="YE"),
                dtype="datetime64[M]",
            ),
        )

    @classmethod
    def get_all_databases(cls) -> list:
        """获取所有QLib数据库名称"""
        return list({
            cls.feature_daily().db_name,
            cls.instrument().db_name,
            cls.calendar().db_name
        })