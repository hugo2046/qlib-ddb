"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2025-02-20 10:44:56
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2025-02-20 17:24:15
Description: ddb连接mysql
"""

from typing import Dict, Optional
from urllib.parse import urlparse, unquote

import pandas as pd
from pydantic import BaseModel, Field, field_validator, validate_arguments

from .ddb_client import DDBClient, DDBConnectionSpec
from .ddb_operator import DDBTableOperator
from .schemas import QlibTableSchema
from .utils import convert_wind_date_to_datetime

class MySQLConnectionSpec(BaseModel):
    """适配多种MySQL方言的连接参数规范"""

    uri: str = Field(..., pattern=r"^mysql(\+[a-z0-9-]+)?://")  # 允许方言扩展

    @field_validator("uri")
    @classmethod
    def validate_and_parse_uri(cls, v: str) -> dict:
        """
        验证并解析MySQL URI，支持以下格式：
        - mysql://user:pass@host:port/db
        - mysql+pymysql://...
        - mysql+mysqldb://...
        """
        parsed = urlparse(unquote(v))

        # 基础验证
        if not parsed.hostname:
            raise ValueError("必须提供有效的主机地址")
        if not parsed.path or len(parsed.path.split("/")) < 2:
            raise ValueError("必须指定数据库名称")

        # 构造连接参数
        config = {
            "host": parsed.hostname,
            "port": parsed.port or 3306,
            "user": parsed.username or "root",
            "password": parsed.password or "",
            "database": parsed.path.split("/")[1],
            "ssl_disabled": True,  # 默认关闭SSL
        }

        # 处理SSL等额外参数
        if "ssl_mode" in parsed.query:
            config.update(ssl_mode=parse_qs(parsed.query)["ssl_mode"][0])
            config["ssl_disabled"] = False

        return config  # 返回可直接用于mysql-connector的参数


class DDBMySQLBridge:
    def __init__(self, ddb_uri: str, mysql_uri: str) -> None:

        config = DDBConnectionSpec(uri=ddb_uri)
        connector = DDBClient(config)

        self.ddb_operator = DDBTableOperator(connector)
        self.ddb_session = self.ddb_operator.conn_mgr.session

        self.spec = MySQLConnectionSpec(uri=mysql_uri)

         # 构建连接表达式
        uri = self.spec.uri
        params = [
            f"'{uri['host']}'", 
            str(uri['port']), 
            f"'{uri['user']}'", 
            f"'{uri['password']}'"
        ]
        if 'database' in uri:
            params.append(f"'{uri['database']}'")
        
        self.connector_expr = f"mysql::connect({', '.join(params)})"

        # NOTE:此时mysql_conn被创建,该mysql连接仅存在于当前节点中
        # 需要dolphinDB有MySQL插件,如果没有，运行load_mysql_plugin方法先行安装
        self.ddb_session.run(
            f"""
        use mysql;
        mysql_conn={self.connector_expr};
        """
        )

    def load_mysql_plugin(self) -> None:
        """在dolphinDB中加载MySQL插件"""
        expr: str = """
        installPlugin("lgbm")
        loadPlugin("lgbm")
        """
        self.ddb_session.run(expr)

    def show_tables(self) -> pd.DataFrame:

        return self.ddb_session.run("mysql::showTables(mysql_conn);")

    def extract_table_schema(self, table_name: str) -> pd.DataFrame:
        expr: str = f""" 
        mysql::extractSchema(mysql_conn, "{table_name}");
        """
        return self.ddb_session.run(expr)

    def close(self) -> None:
        """
        关闭当前连接
        """
        self.ddb_session.run("mysql::close(mysql_conn)")

    @validate_arguments
    def load_table(
        self,
        table_or_query: str,
        table_schema: Optional[Dict[str, str]] = None,
        start_row: Optional[int] = None,
        row_num: Optional[int] = None,
        allow_empty_table: bool = False,
    ) -> pd.DataFrame:
        """
        从MySQL加载数据到DolphinDB内存表

        :param table_or_query: MySQL表名或查询语句（需带引号）
        :param table_schema: 列类型映射（列名 -> 类型字符串）
        :param start_row: 起始行号（仅对表名有效）
        :param row_num: 读取行数（仅对表名有效）
        :param allow_empty_table: 是否允许加载空表
        :return: 加载后的DolphinDB表引用

        示例：
        >>> bridge.load_table(
                "stock_data",
                schema={"symbol": "SYMBOL", "price": "DOUBLE"},
                start_row=100,
                row_num=500
            )
        """
        # 参数校验
        if not isinstance(table_or_query, str):
            raise TypeError("table_or_query必须是字符串类型")

        if start_row is not None and start_row < 0:
            raise ValueError("start_row不能为负数")

        # 构建参数映射
        params = [
            "mysql_conn",  # conn
            f"""'{table_or_query}'""",  # tableOrQuery
            self._process_schema(table_schema) if table_schema else "",  # schema
            str(start_row) if start_row is not None else "",  # startRow
            str(row_num) if row_num is not None else "",  # rowNum
            "true" if allow_empty_table else "false",  # allowEmptyTable
        ]

        # 生成参数字符串
        param_str = ", ".join(params)
        expr = f"""mysql::load({param_str})"""
        try:
            return self.ddb_session.run(expr)
        except Exception as e:
            raise RuntimeError(f"加载MySQL表失败: {str(e)}") from e

    def _process_schema(self, table_schema: Optional[Dict[str, str]]) -> Optional[str]:
        """将schema字典转换为DolphinDB需要的表格式"""
        if not table_schema:
            return None

        if not isinstance(table_schema, dict):
            raise TypeError("schema必须是字典类型")

        self.ddb_session.run(
            f"""
            schema = table({list(table_schema.keys())} as name, {list(table_schema.values())} as type)
            """
        )
        return "schema"


def init_qlib_ddb_from_mysql(ddb_uri: str, mysql_uri: str) -> None:
    """
    从MySQL初始化QLib所需的DolphinDB数据库。

    该函数将MySQL中的数据写入DolphinDB，用于QLib的数据初始化。
    它创建并填充了三个主要表：Instrument、Calendar和FeatureDaily。

    :param ddb_uri: DolphinDB的连接URI
    :type ddb_uri: str
    :param mysql_uri: MySQL的连接URI
    :type mysql_uri: str
    :return: None

    .. note::
        此函数使用DDBMySQLBridge类来处理数据库间的连接和数据传输。
    """
    bridge = DDBMySQLBridge(ddb_uri, mysql_uri)

    def _extract_columns(schema_func):
        """
        从给定的schema函数中提取列信息。

        :param schema_func: 返回表结构的函数
        :type schema_func: callable
        :return: 所有列名的字符串和列名到类型的字典映射
        :rtype: tuple
        """
        cols = schema_func().columns
        all_cols = ",".join(col[0] for col in cols)
        name_type = dict(cols)
        return all_cols, name_type
    
    def _sync_table(schema_func, table_name, where_clause=""):
        """
        同步指定的MySQL表到DolphinDB。

        :param schema_func: 返回表结构的函数
        :type schema_func: callable
        :param table_name: MySQL中的表名
        :type table_name: str
        :param where_clause: SQL WHERE子句（可选）
        :type where_clause: str
        """
        print(f"正在同步 {schema_func.__name__} 从 {table_name}...")
        cols, name_type = _extract_columns(schema_func)
        query = f"SELECT {cols} FROM {table_name}" + (
            f" WHERE {where_clause}" if where_clause else ""
        )

        data = bridge.load_table(query)

        convert_wind_date_to_datetime(data, name_type, inplace=True)

        bridge.ddb_operator.table_appender(
            schema_func().db_name, schema_func().table_name, data
        )
        print(f"已完成 {schema_func.__name__} 从 {table_name} 的同步。\n")

    _sync_table(QlibTableSchema.instrument, "ASHAREDESCRIPTION")
    _sync_table(QlibTableSchema.calendar, "ASHARECALENDAR", 'S_INFO_EXCHMARKET="SSE"')
    _sync_table(
        QlibTableSchema.feature_daily,
        "ASHAREEODPRICES",
        "TRADE_DT BETWEEN 20210101 AND 20231231",
    )


# def init_qlib_ddb_from_mysql(ddb_uri: str, mysql_uri: str) -> None:
#     """
#     从MySQL初始化QLib所需的DolphinDB数据库。
#     该函数将MySQL中的数据写入DolphinDB，用于QLib的数据初始化。
#     它创建并填充了三个主要表：Instrument、Calendar和FeatureDaily。

#     :param ddb_uri: DolphinDB的连接URI
#     :param mysql_uri: MySQL的连接URI
#     :return: None

#     .. note::
#         此函数使用DolphinDB的原生脚本语法执行数据转换和加载操作。
#     """
#     bridge = DDBMySQLBridge(ddb_uri, mysql_uri)

#     def _extract_columns(schema_func):
#         cols = schema_func().columns
#         all_cols = ",".join(col[0] for col in cols)
#         name_type = dict(cols)
#         partition_columns = schema_func().partition_columns
#         return all_cols, name_type,partition_columns

#     instrument_cols, instrument_name_type, instrument_partition_columns = _extract_columns(QlibTableSchema.instrument)
#     calendar_cols, calendar_name_type,calendar_partition_columns = _extract_columns(QlibTableSchema.calendar)
#     feature_daily_cols, feature_daily_name_type,feature_daily_partition_columns = _extract_columns(
#         QlibTableSchema.feature_daily
#     )

#     bridge.ddb_session.upload(
#         {
#             "instrument_name_type": instrument_name_type,
#             "calendar_name_type": calendar_name_type,
#             "feature_daily_name_type": feature_daily_name_type,
#         }
#     )

#     # 定义DolphinDB脚本
#     ddb_scripts = {
#         "convert_function": """
#         def convert_winddate_type_to_ddb(mutable tb, name_type) {
#             for (col in name_type.keys()) {
#                 if (name_type[col] == DATE) {
#                     tb.replaceColumn!(col, temporalParse(tb[col]$STRING, "yyyy.MM.dd"));
#                 } else if (name_type[col] == DOUBLE) {
#                     tb.replaceColumn!(col, tb[col]$DOUBLE);
#                 }
#             }
#         }
#         """,
#         "sync_instrument": f"""
#         QlibInstrumentDB = database("dfs://QlibInstrument");
#         mysql::loadEx(mysql_conn, QlibInstrumentDB, `Instrument,`{instrument_partition_columns},
#             "select {instrument_cols} from ASHAREDESCRIPTION",,,,convert_winddate_type_to_ddb{{,instrument_name_type}});
#         """,
#         "sync_calendar": f"""
#         QlibCalendarDB = database("dfs://QlibCalendar");
#         mysql::loadEx(mysql_conn, QlibCalendarDB, `Calendar,`{calendar_partition_columns},
#             "select {calendar_cols} from ASHARECALENDAR WHERE S_INFO_EXCHMARKET='SZSE'",,,,convert_winddate_type_to_ddb{{,calendar_name_type}});
#         """,
#         "sync_feature_daily": f"""
#         QlibFeatureDailyDB = database("dfs://QlibFeature");
#         mysql::loadEx(mysql_conn, QlibFeatureDailyDB, `FeatureDaily,`{feature_daily_partition_columns},
#             "select {feature_daily_cols} from ASHAREEODPRICES WHERE TRADE_DT BETWEEN 20210101 AND 20241231",,,,convert_winddate_type_to_ddb{{,feature_daily_name_type}});
#         """,
#     }

#     # 执行DolphinDB脚本
#     for script_name, script in ddb_scripts.items():
#         try:
#             bridge.ddb_session.run(script)
#             print(f"成功执行脚本: {script_name}")
#         except Exception as e:
#             print(f"执行脚本 {script_name} 时出错: {str(e)}")

#     bridge.close()
