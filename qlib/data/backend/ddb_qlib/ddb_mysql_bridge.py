"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2025-02-20 10:44:56
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2025-02-20 17:24:15
Description: ddb连接mysql
"""

from typing import Dict, Optional,List
from urllib.parse import urlparse, unquote, parse_qs

import pandas as pd
from pydantic import BaseModel, Field, field_validator, validate_arguments

from .ddb_client import DDBClient, DDBConnectionSpec
from .ddb_operator import DDBTableOperator
from .schemas import QlibTableSchema, FIELDS_MAPPING
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
        """
        初始化DDBMySQLBridge实例
        
        :param ddb_uri: DolphinDB连接URI
        :param mysql_uri: MySQL连接URI
        """
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

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，确保连接被正确关闭"""
        self.close()
        # 不抑制任何异常
        return False

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
        try:
            self.ddb_session.run("mysql::close(mysql_conn)")
        except Exception as e:
            # 即使关闭连接失败，也不应该中断程序执行
            print(f"警告: 关闭MySQL连接时出现错误: {str(e)}")

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
            f'"{table_or_query}"',  # tableOrQuery
            self._process_schema(table_schema) if table_schema else " ",  # schema
            str(start_row) if start_row is not None else " ",  # startRow
            str(row_num) if row_num is not None else " ",  # rowNum
            "true" if allow_empty_table else "false",  # allowEmptyTable
        ]

        # 生成参数字符串，过滤掉空参数
        param_str = ", ".join([p for p in params if p])
        expr = f"""mysql::load({param_str})"""
        try:
            result = self.ddb_session.run(expr)
            if result is None and not allow_empty_table:
                raise RuntimeError("加载MySQL表返回空结果")
            return result
        except Exception as e:
            raise RuntimeError(f"加载MySQL表失败: {str(e)}") from e

    def _process_schema(self, table_schema: Optional[Dict[str, str]]) -> Optional[str]:
        """
        将schema字典转换为DolphinDB需要的表格式
        
        :param table_schema: 列名到类型的映射字典
        :return: DolphinDB中的schema表名，如果输入为空则返回None
        """
        if not table_schema:
            return None

        if not isinstance(table_schema, dict):
            raise TypeError("schema必须是字典类型")

        # 验证schema中的值是否为字符串类型
        for key, value in table_schema.items():
            if not isinstance(value, str):
                raise TypeError(f"schema中的值必须是字符串类型，但'{key}'的值为{type(value)}")

        try:
            self.ddb_session.run(
                f"""
                schema = table({list(table_schema.keys())} as name, {list(table_schema.values())} as type)
                """
            )
            return "schema"
        except Exception as e:
            raise RuntimeError(f"处理schema失败: {str(e)}") from e


class QlibDDBMySQLInitializer:
    """
    QLib DolphinDB MySQL初始化器类
    
    该类用于从MySQL初始化QLib所需的DolphinDB数据库。
    支持单独同步每个表或者全部同步。
    """
    
    def __init__(self, ddb_uri: str, mysql_uri: str):
        """
        初始化QlibDDBMySQLInitializer实例
        
        :param ddb_uri: DolphinDB的连接URI
        :type ddb_uri: str
        :param mysql_uri: MySQL的连接URI
        :type mysql_uri: str
        """
        self.ddb_uri = ddb_uri
        self.mysql_uri = mysql_uri
        self._bridge = None
    
    def __enter__(self):
        """上下文管理器入口"""
        self._bridge = DDBMySQLBridge(self.ddb_uri, self.mysql_uri)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        if self._bridge:
            try:
                self._bridge.close()
            except:
                pass
        return False
    
    def _get_bridge(self):
        """获取bridge实例，如果不存在则创建"""
        if self._bridge is None:
            try:
                self._bridge = DDBMySQLBridge(self.ddb_uri, self.mysql_uri)
            except Exception as e:
                raise RuntimeError(f"初始化DDBMySQLBridge失败: {str(e)}") from e
        return self._bridge
    
    def _extract_columns(self, schema_func):
        """
        从给定的schema函数中提取列信息。

        :param schema_func: 返回表结构的函数
        :type schema_func: callable
        :return: 所有列名的字符串和列名到类型的字典映射
        :rtype: tuple
        """
        try:
            schema = schema_func()
            all_cols = ",".join(schema.map_columns_to_fields())
            name_type = dict(schema.columns)
            return all_cols, name_type
        except Exception as e:
            raise RuntimeError(f"提取列信息失败: {str(e)}") from e
    
    def _sync_table(self, schema_func, table_name, where_clause=""):
        """
        同步指定的MySQL表到DolphinDB。

        :param schema_func: 返回表结构的函数
        :type schema_func: callable
        :param table_name: MySQL中的表名
        :type table_name: str
        :param where_clause: SQL WHERE子句（可选）
        :type where_clause: str
        """
        bridge = self._get_bridge()
        try:
            print(f"正在同步 {schema_func.__name__} 从 {table_name}...")
            cols, name_type = self._extract_columns(schema_func)
            query = f"SELECT {cols} FROM {table_name}" + (
                f" WHERE {where_clause}" if where_clause else ""
            )

            data = bridge.load_table(query)
            if table_name in "ASHAREEODPRICES":
                data = data.rename(columns=FIELDS_MAPPING)
            convert_wind_date_to_datetime(data, name_type, inplace=True)

            bridge.ddb_operator.table_appender(
                schema_func().db_name, schema_func().table_name, data
            )
            print(f"已完成 {schema_func.__name__} 从 {table_name} 的同步。\n")
        except Exception as e:
            raise RuntimeError(f"同步表 {table_name} 失败: {str(e)}") from e
    
    def sync_instrument(self):
        """
        同步Instrument表（ASHAREDESCRIPTION）
        """
        self._sync_table(QlibTableSchema.instrument, "ASHAREDESCRIPTION")
    
    def sync_calendar(self, exchange_market: str = "SSE"):
        """
        同步Calendar表（ASHARECALENDAR）
        
        :param exchange_market: 交易所市场，默认为"SSE"
        :type exchange_market: str
        """
        where_clause = f'S_INFO_EXCHMARKET="{exchange_market}"'
        self._sync_table(QlibTableSchema.calendar, "ASHARECALENDAR", where_clause)
    
    def sync_feature_daily(self, start_date: str = "20100101", end_date: str = "20241231"):
        """
        同步FeatureDaily表（ASHAREEODPRICES）
        
        :param start_date: 开始日期，格式：YYYYMMDD
        :type start_date: str
        :param end_date: 结束日期，格式：YYYYMMDD
        :type end_date: str
        """
        where_clause = f"TRADE_DT BETWEEN {start_date} AND {end_date}"
        self._sync_table(QlibTableSchema.feature_daily, "ASHAREEODPRICES", where_clause)
    
    def sync_index_daily(self, index_codes: List[str], start_date: str = "20100101", end_date: str = "20241231"):
        """
        同步指数日线数据到专门的IndexDaily表
        
        :param index_codes: 指数代码列表或单个指数代码
        :type index_codes: List[str] 或 str
        :param start_date: 开始日期，格式：YYYYMMDD
        :type start_date: str
        :param end_date: 结束日期，格式：YYYYMMDD
        :type end_date: str
        :raises RuntimeError: 当同步失败时抛出异常
        """
        # 参数标准化：确保index_codes为列表
        if isinstance(index_codes, str):
            index_codes = [index_codes]
        elif not isinstance(index_codes, list):
            raise TypeError("index_codes必须是字符串或字符串列表")
        
        if not index_codes:
            raise ValueError("index_codes不能为空")
        
        # 构建WHERE条件
        if len(index_codes) == 1:
            select_index_code = f"S_INFO_WINDCODE = '{index_codes[0]}'"
        else:
            # 为每个代码添加引号
            quoted_codes = [f"'{code}'" for code in index_codes]
            select_index_code = f"S_INFO_WINDCODE IN ({','.join(quoted_codes)})"
            
        bridge = self._get_bridge()
        try:
            print(f"正在同步指数 {','.join(index_codes)} 从 AINDEXEODPRICES 至 IndexDaily 表...")
            
            # 定义指数列映射
            index_columns = (
                "TRADE_DT as date, "
                "S_INFO_WINDCODE as code, "
                "S_DQ_OPEN as open, "
                "S_DQ_HIGH as high, "
                "S_DQ_LOW as low, "
                "S_DQ_CLOSE as close, "
                "S_DQ_VOLUME as volume, "
                "S_DQ_AMOUNT as amount, "
                "0 as factor, "
                "0 as vwap"
            )
            
            query = (
                f"SELECT {index_columns} FROM AINDEXEODPRICES "
                f"WHERE TRADE_DT BETWEEN {start_date} AND {end_date} "
                f"AND {select_index_code}"
            )

            data = bridge.load_table(query)
            
            if data.empty:
                print(f"警告: 指数 {','.join(index_codes)} 在时间范围 {start_date}-{end_date} 内无数据")
                return
           
            # 转换日期格式
            convert_wind_date_to_datetime(data, {"date": "DATE"}, inplace=True)

            # 获取指数表信息
            features_schema = QlibTableSchema.feature_daily()
            bridge.ddb_operator.table_appender(
                features_schema.db_name, 
                features_schema.table_name, 
                data
            )
            
            print(f"已完成指数数据同步：{len(data)} 条记录已写入 {features_schema.table_name} 表\n")
            
        except Exception as e:
            error_msg = f"同步指数表失败 - 指数: {','.join(index_codes)}, 时间范围: {start_date}-{end_date}"
            raise RuntimeError(f"{error_msg}: {str(e)}") from e
    
    def sync_all(self, exchange_market: str = "SSE", start_date: str = "20100101", end_date: str = "20241231"):
        """
        同步所有表
        
        :param exchange_market: 交易所市场，默认为"SSE"
        :type exchange_market: str
        :param start_date: 开始日期，格式：YYYYMMDD
        :type start_date: str
        :param end_date: 结束日期，格式：YYYYMMDD
        :type end_date: str
        """
        try:
            self.sync_instrument()
            self.sync_calendar(exchange_market)
            self.sync_feature_daily(start_date, end_date)
        except Exception as e:
            raise RuntimeError(f"同步所有表失败: {str(e)}") from e
    
    def close(self):
        """关闭连接"""
        if self._bridge:
            try:
                self._bridge.close()
            except:
                pass
            self._bridge = None


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
        此函数为向后兼容性保留，建议使用QlibDDBMySQLInitializer类。
    """
    with QlibDDBMySQLInitializer(ddb_uri, mysql_uri) as initializer:
        initializer.sync_all()



