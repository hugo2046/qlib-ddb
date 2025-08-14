"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2025-02-20 13:57:41
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2025-02-20 14:32:51
Description: 数据库操作
"""

from typing import Dict, List, Optional, Tuple, Union

import dolphindb as ddb
import pandas as pd

from .ddb_client import DDBClient, DDBConnectionSpec
from .schemas import QlibTableSchema, TableSchema


class DDBTableOperator:
    """适配单会话连接的操作类"""

    def __init__(self, conn_mgr: DDBClient):
        self.conn_mgr = conn_mgr

    def _normalize_db_path(self, name: str) -> str:
        """路径标准化方法"""
        return f"dfs://{name}" if not name.startswith("dfs://") else name

    def exist_database(self, db_name: str) -> bool:
        """检查数据库存在性"""
        session = self.conn_mgr.session
        db_path = self._normalize_db_path(db_name)
        return session.existsDatabase(db_path)

    def exist_table(self, db_name: str, table_name: str) -> bool:
        """检查表存在性"""
        session = self.conn_mgr.session
        db_path = self._normalize_db_path(db_name)
        return session.existsTable(db_path, table_name)

    def create_database(
        self,
        db_name: str,
        partition_type: int,
        partitions: int,
        engine: str = "OLAP",
        *,
        session: Optional[ddb.Session] = None,
    ) -> None:
        """创建数据库"""
        if session is None:
            session: ddb.Session = self.conn_mgr.session

        db_path = self._normalize_db_path(db_name)
        return session.database(
            dbName=db_name,
            partitionType=partition_type,
            partitions=partitions,
            dbPath=db_path,
            engine=engine,
        )

    def create_partitioned_table(
        self,
        db_name: str,
        table_name: str,
        schema: List[Tuple[str, str]],
        partition_type: int,
        partitions,
        *,
        partition_columns: Union[str, List[str]],
        sort_columns: Union[str, List[str]] = None,
        engine: str = "OLAP",
        keep_duplicates: str = None,
        sort_key_mapping_function: Union[str, List[str]] = None,
        primary_key: Union[str, List[str]] = None,
        soft_delete: bool = None,
        indexes: Dict[str, Union[str, List[str]]] = None,
        session: Optional[ddb.Session] = None,
    ) -> None:
        """
        创建分区表。

        :param db_name: 数据库名称
        :type db_name: str
        :param table_name: 表名
        :type table_name: str
        :param schema: 表结构，列表中每个元组包含列名和列类型
        :type schema: List[Tuple[str, str]]
        :param partition_type: 分区类型
        :type partition_type: int
        :param partitions: 分区方案
        :param partition_columns: 分区列
        :type partition_columns: Union[str, List[str]]
        :param sort_columns: 排序列，仅在TSDB引擎下有效，默认为None
        :type sort_columns: Union[str, List[str]], optional
        :param engine: 存储引擎，默认为"OLAP"
        :type engine: str, optional
        :param keep_duplicates: 重复值处理方式，默认为None
        :type keep_duplicates: str, optional
        :param sort_key_mapping_function: 排序键映射函数，默认为None
        :type sort_key_mapping_function: Union[str, List[str]], optional
        :param primary_key: 主键，默认为None
        :type primary_key: Union[str, List[str]], optional
        :param soft_delete: 是否启用软删除，默认为None
        :type soft_delete: bool, optional
        :param indexes: 索引，默认为None
        :type indexes: Dict[str, Union[str, List[str]]], optional
        :param session: 会话对象，默认为None
        :type session: ddb.Session, optional
        :raises ValueError: 如果表已存在则抛出异常

        :注意:
            - partitions的数据类型需要与partition_columns列的类型一致。
            - 例如，如果partition_columns为trade_dt且类型为DATE，那么partitions也需要为datetime的类似格式。
        """

        # 添加session参数方便创建COMPO时使用；否则外面传入的partitions=[db1,db2]时内部收不到
        if session is None:
            session: ddb.Session = self.conn_mgr.session
        db_path = self._normalize_db_path(db_name)

        if self.exist_table(db_name, table_name):
            raise ValueError(f"{db_path}/{table_name}已存在!")

        if not self.exist_database(db_path):

            db = self.create_database(
                db_name, partition_type, partitions, engine, session=session
            )
        else:
            db = session.database(dbPath=db_path)
        
        # 创建临时的schema表
        col_names = "".join([f"`{col[0]}" for col in schema])
        col_types = "".join([f"`{col[1]}" for col in schema])
        
        schema_expr = (
            f"schema_t = table(1:0, {col_names}, {col_types})"
            if len(schema) > 1
            else f'schema_t = table(1:0,["{schema[0][0]}"], ["{schema[0][1]}"])'
        )
        session.run(schema_expr)

        db.createPartitionedTable(
            table=session.table(data="schema_t"),
            tableName=table_name,
            partitionColumns=partition_columns,
            sortColumns=sort_columns,
            keepDuplicates=keep_duplicates,
            sortKeyMappingFunction=sort_key_mapping_function,
            primaryKey=primary_key,
            softDelete=soft_delete,
            indexes=indexes,
        )
        # 清理临时表
        session.run("undef(`schema_t)")

    def table_appender(self, db_name: str, table_name: str, data: pd.DataFrame) -> None:
        """
        将数据追加到指定的表中。

        :param db_name: 数据库名称
        :type db_name: str
        :param table_name: 表名
        :type table_name: str
        :param data: 要追加的数据
        :type data: pd.DataFrame
        :raises ValueError: 如果指定的表不存在
        """
        session = self.conn_mgr.session
        db_path = self._normalize_db_path(db_name)

        if not self.exist_table(db_name, table_name):
            raise ValueError(f"{db_path}/{table_name}不存在!")

        table = session.loadTable(table_name, db_path)
        table_cols = table.schema["name"].tolist()
    
        # 确保数据列名顺序与表列名顺序一致
        data = data.reindex(columns=table_cols)

        appender =ddb.tableAppender(tableName=table_name, ddbSession=session, dbPath=db_path)
        appender.append(data)

    def table_upsert(
        self,
        db_name: str,
        table_name: str,
        data: pd.DataFrame,
        key_col_names: Optional[Union[str, List[str]]] = None,
        sort_columns: Optional[Union[str, List[str]]] = None,
    ) -> None:
        """
        将数据帧中的数据插入或更新到指定的数据库表中。

        :param db_name: 数据库名称
        :type db_name: str
        :param table_name: 表名
        :type table_name: str
        :param data: 包含要插入或更新的数据的数据帧
        :type data: pd.DataFrame
        :param key_col_names: 用于确定唯一记录的列名，可以是单个字符串或字符串列表
        :type key_col_names: Optional[Union[str, List[str]]]
        :param sort_columns: 用于排序的列名，可以是单个字符串或字符串列表
        :type sort_columns: Optional[Union[str, List[str]]]
        :raises ValueError: 如果指定的表不存在
        :return: 无返回值
        :rtype: None
        """
        session = self.conn_mgr.session
        db_path = self._normalize_db_path(db_name)

        if not self.exist_table(db_name, table_name):
            raise ValueError(f"表 {db_path}/{table_name} 不存在!")

        table = session.loadTable(table_name, db_path)
        table_cols = table.schema["name"].tolist()

        # 确保数据列名顺序与表列名顺序一致
        data = data.reindex(columns=table_cols)

        # 统一处理 key_col_names 和 sort_columns
        key_col_names = self._normalize_column_names(key_col_names)
        sort_columns = self._normalize_column_names(sort_columns)

        upserter = ddb.tableUpsert(
            tableName=table_name,
            ddbSession=session,
            dbPath=db_path,
            keyColNames=key_col_names,
            sortColumns=sort_columns,
        )
        upserter.upsert(data)

    @staticmethod
    def _normalize_column_names(columns: Optional[Union[str, List[str]]]) -> List[str]:
        """
        将列名标准化为列表格式。

        :param columns: 输入的列名，可以是None、字符串或字符串列表
        :type columns: Optional[Union[str, List[str]]]
        :return: 标准化后的列名列表
        :rtype: List[str]
        """
        if columns is None:
            return []
        elif isinstance(columns, str):
            return [columns]
        elif isinstance(columns, list):
            return columns
        else:
            raise ValueError("列名必须是字符串或字符串列表")


def create_table(
    uri: str,
    db_name: str,
    table_name: str,
    schema: tuple,
    partition_type,
    partitions,
    partition_columns: str,
    engine: str,
    primary_key: str = None,
) -> None:
    """
    通用创建表函数

    此函数用于在DolphinDB中创建分区表。

    :param uri: DolphinDB连接URI
    :type uri: str
    :param db_name: 数据库名称
    :type db_name: str
    :param table_name: 表名
    :type table_name: str
    :param schema: 表结构，包含列名和列类型的元组
    :type schema: tuple
    :param partition_type: 分区类型
    :type partition_type: int
    :param partitions: 分区方案
    :type partitions: Any
    :param partition_columns: 分区列名
    :type partition_columns: str
    :param engine: 存储引擎
    :type engine: str
    :param primary_key: 主键列名，默认为None
    :type primary_key: str, optional

    :return: 无返回值
    :rtype: None

    :raises: 可能抛出的异常包括连接错误、表已存在等

    .. note::
       确保在调用此函数之前已正确配置DolphinDB连接参数。
    """
    config = DDBConnectionSpec(uri=uri)
    connector = DDBClient(config)
    db_accessor = DDBTableOperator(connector)

    db_accessor.create_partitioned_table(
        db_name,
        table_name,
        schema,
        partition_type,
        partitions,
        partition_columns=partition_columns,
        engine=engine,
        primary_key=primary_key,
    )
    print(f"{db_name}/{table_name}生成创建完毕!")


def create_qlib_table(uri: str, schema: TableSchema) -> None:
    """通用表创建函数"""
    create_table(
        uri=uri,
        db_name=schema.db_name,
        table_name=schema.table_name,
        schema=schema.columns,
        partition_type=schema.partition_type,
        partitions=schema.partitions,
        partition_columns=schema.partition_columns,
        engine=schema.engine,
        primary_key=schema.primary_key,
    )


def create_feature_daily_table(uri: str) -> None:
    schema = QlibTableSchema.feature_daily()
    create_qlib_table(uri, schema)


def create_calendar_table(uri: str) -> None:
    schema = QlibTableSchema.calendar()
    create_qlib_table(uri, schema)


def create_instrument_table(uri: str, table_name: str = "ashares") -> None:
    schema = QlibTableSchema.instrument(table_name)
    create_qlib_table(uri, schema)


def clean_qlib_db(uri: str) -> None:
    """清理Qlib数据库"""
    config = DDBConnectionSpec(uri=uri)
    connector = DDBClient(config)

    session = connector.session

    db_names = QlibTableSchema.get_all_databases()
    expr_lines = [
        f'try{{ dropDatabase("dfs://{db}") }} catch(ex) {{}}'
        for db in db_names
    ]
    expr = "\n".join(expr_lines)
    
    session.run(expr)


def write_df_to_ddb(
    db_name: str,
    table_name: str,
    data: pd.DataFrame,
    upsert: bool = False,
    key_col_names: Optional[Union[str, List[str]]] = None,
    sort_columns: Optional[Union[str, List[str]]] = None,
    uri: Optional[str] = None,
) -> None:
    """
    将DataFrame数据写入DolphinDB表。

    :param db_name: 数据库名称
    :type db_name: str
    :param table_name: 表名
    :type table_name: str
    :param data: 需要写入的DataFrame数据
    :type data: pd.DataFrame
    :param upsert: 是否使用upsert方式写入，默认为False（追加），True则使用upsert
    :type upsert: bool
    :param key_col_names: upsert时用于唯一性判断的列名
    :type key_col_names: Optional[Union[str, List[str]]]
    :param sort_columns: upsert时用于排序的列名
    :type sort_columns: Optional[Union[str, List[str]]]
    :param conn_mgr: DDBClient连接管理器，默认为None（需外部提前初始化）
    :type conn_mgr: Optional[DDBClient]
    """
    
    
    if uri is None:
        raise ValueError("请传入已初始化的DDBClient实例(conn_mgr)")

    config = DDBConnectionSpec(uri=uri)
    connector = DDBClient(config)
    operator = DDBTableOperator(connector)
    if upsert:
        operator.table_upsert(
            db_name=db_name,
            table_name=table_name,
            data=data,
            key_col_names=key_col_names,
            sort_columns=sort_columns,
        )
    else:
        operator.table_appender(
            db_name=db_name,
            table_name=table_name,
            data=data,
        )
        
        
def import_instruments_csv_to_ddb(
    data_or_path: Union[str, pd.DataFrame],
    db_name: str,
    table_name: str,
    uri: str,
    windcode_convert: bool = True,
    sep: str = "\t",
    header: Union[int, None] = None,
    col_names: List[str] = ["S_INFO_WINDCODE", "S_INFO_LISTDATE", "S_INFO_DELISTDATE"],
) -> None:
    """
    从CSV文件或DataFrame读取成分股数据并写入DolphinDB表。

    :param data_or_path: CSV文件路径或DataFrame
    :type data_or_path: str or pd.DataFrame
    :param db_name: DolphinDB数据库名
    :type db_name: str
    :param table_name: DolphinDB表名
    :type table_name: str
    :param uri: DolphinDB连接URI
    :type uri: str
    :param windcode_convert: 是否将windcode转换为"000001.SZ"格式，默认为True
    :type windcode_convert: bool
    :param sep: CSV分隔符，默认为制表符
    :type sep: str
    :param header: CSV文件头，默认为None
    :type header: int or None
    :param col_names: 列名列表，默认为["S_INFO_WINDCODE", "S_INFO_LISTDATE", "S_INFO_DELISTDATE"]
    :type col_names: List[str]
    """

    try:
        # 判断输入类型
        if isinstance(data_or_path, str):
            df = pd.read_csv(data_or_path, sep=sep, header=header)
            df.columns = col_names
        elif isinstance(data_or_path, pd.DataFrame):
            df = data_or_path.copy()
            # 如果列名不一致，尝试重命名
            if list(df.columns) != col_names:
                df.columns = col_names
        else:
            raise ValueError("data_or_path 必须为文件路径字符串或DataFrame对象")

        # 可选：转换windcode格式为"000001.SZ"
        if windcode_convert:
            df["S_INFO_WINDCODE"] = df["S_INFO_WINDCODE"].apply(
                lambda x: f"{x[-6:]}.{x[:2]}"
            )

        # 转换日期列为datetime64[ns]
        for col in ["S_INFO_LISTDATE", "S_INFO_DELISTDATE"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # 写入DolphinDB
        write_df_to_ddb(db_name, table_name, df, uri=uri)
    except Exception as e:
        raise ValueError("Failed to import instruments data to DolphinDB") from e
    
    
def import_calendar_csv_to_ddb(
    data_or_path: Union[str, pd.DataFrame],
    db_name: str,
    table_name: str,
    uri: str,
    col_name: str = "TRADE_DAYS",
) -> None:
    """
    从CSV文件或DataFrame读取交易日历数据并写入DolphinDB表。

    :param data_or_path: CSV文件路径或DataFrame
    :type data_or_path: str or pd.DataFrame
    :param db_name: DolphinDB数据库名
    :type db_name: str
    :param table_name: DolphinDB表名
    :type table_name: str
    :param uri: DolphinDB连接URI
    :type uri: str
    :param col_name: 日历列名，默认为"TRADE_DAYS"
    :type col_name: str
    """

    try:
        # 判断输入类型
        if isinstance(data_or_path, str):
            df = pd.read_csv(data_or_path)
        elif isinstance(data_or_path, pd.DataFrame):
            df = data_or_path.copy()
        else:
            raise ValueError("data_or_path 必须为文件路径字符串或DataFrame对象")

        # 检查是否包含指定列
        if col_name not in df.columns:
            raise ValueError(f"{col_name} column is missing")

        # 转换为datetime类型
        df[col_name] = pd.to_datetime(df[col_name], errors="coerce")
        # 只保留日历列
        df = df[[col_name]]

        # 写入DolphinDB
        write_df_to_ddb(db_name=db_name, table_name=table_name, data=df, uri=uri)
    except Exception as e:
        raise ValueError("Failed to import calendar data to DolphinDB") from e
    
def import_features_csv_to_ddb(
    data_or_path: Union[str, pd.DataFrame],
    db_name: str,
    table_name: str,
    uri: str,
    col_mapping: Optional[Dict[str, str]] = None,
    upsert: bool = False,
) -> None:
    """
    从CSV文件或DataFrame读取特征数据并写入DolphinDB表。

    :param data_or_path: CSV文件路径或DataFrame
    :type data_or_path: str or pd.DataFrame
    :param db_name: DolphinDB数据库名
    :type db_name: str
    :param table_name: DolphinDB表名
    :type table_name: str
    :param uri: DolphinDB连接URI
    :type uri: str
    :param col_mapping: 列名映射，默认为FEATURES_TABLE_MAPPING
    :type col_mapping: Optional[Dict[str, str]]
    :param upsert: 是否upsert写入，默认为False
    :type upsert: bool
    """

    FEATURES_TABLE_MAPPING: Dict = {
        "TRADE_DT": "date",
        "S_INFO_WINDCODE": "code",
        "S_DQ_ADJOPEN": "open",
        "S_DQ_ADJHIGH": "high",
        "S_DQ_ADJLOW": "low",
        "S_DQ_ADJCLOSE": "close",
        "S_DQ_VOLUME": "volume",
        "S_DQ_AMOUNT": "amount",
        "S_DQ_AVGPRICE": "vwap",
        "S_DQ_ADJFACTOR": "factor",
    }

    try:
        # 判断输入类型
        if isinstance(data_or_path, str):
            df = pd.read_csv(data_or_path)
        elif isinstance(data_or_path, pd.DataFrame):
            df = data_or_path.copy()
        else:
            raise ValueError("data_or_path 必须为文件路径字符串或DataFrame对象")

        if df.empty:
            raise ValueError("DataFrame is empty")

        if col_mapping is None:
            col_mapping = FEATURES_TABLE_MAPPING

        # 列名重命名
        df.rename(columns=col_mapping, inplace=True)

        # 检查缺失列
        must_have = ["date", "code", "open", "high", "low", "close", "factor"]
        optional = ["vwap", "amount", "volume"]
        missing_cols = [v for v in must_have if v not in df.columns]
        if missing_cols:
            raise ValueError(f"缺失必要列: {missing_cols}")
        warn_cols = [v for v in optional if v not in df.columns]
        if warn_cols:
            import warnings

            warnings.warn(f"可选列缺失: {warn_cols}")

        # 转换date为datetime类型
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # 写入DolphinDB
        write_df_to_ddb(
            db_name=db_name,
            table_name=table_name,
            data=df,
            uri=uri,
            upsert=upsert,
            key_col_names=["code", "date"] if upsert else None,
        )
    except Exception as e:
        raise ValueError("Failed to import features data to DolphinDB") from e