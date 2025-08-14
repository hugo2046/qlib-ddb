"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2025-02-19 14:29:41
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2025-04-17 16:02:46
Description: 用于连接ddb
"""

from typing import List, Optional, Union
from urllib.parse import unquote, urlparse

import dolphindb as ddb
import pandas as pd
from pydantic import BaseModel, field_validator


class DDBConnectionSpec(BaseModel):
    """单会话专用配置"""

    uri: str  # 格式：dolphindb://user:pass@host:port

    @field_validator("uri")
    @classmethod
    def validate_uri(cls, v: str) -> str:
        parsed = urlparse(unquote(v))
        if parsed.scheme != "dolphindb":
            raise ValueError("协议头必须为 dolphindb://")
        if not parsed.hostname or not parsed.port:
            raise ValueError("必须包含有效的主机名和端口")
        return parsed.geturl()


class DDBClient:
    
    _pool_instance = None  # 类变量用于存储连接池实例

    
    def __init__(self, config: DDBConnectionSpec):
        self._config = config
        self._session = self._create_session()

    def _parse_uri(self) -> tuple[str, str, str, int]:
        """解析连接参数"""
        parsed = urlparse(self._config.uri)
        return (
            parsed.username or "admin",
            parsed.password or "",
            parsed.hostname,
            parsed.port,
        )

    def _create_session(self) -> ddb.Session:
        """创建单会话"""
        user, pwd, host, port = self._parse_uri()
        session = ddb.session(
            protocol=ddb.settings.PROTOCOL_DDB,
            show_output=True,  # 使用DDB协议
            enableASYNC=False,
        )
        session.connect(host=host, port=port, userid=user, password=pwd, reconnect=True)
        return session

    def _create_pool(self,threadNum:int=4) -> ddb.DBConnectionPool:
        """创建连接池"""
        user, pwd, host, port = self._parse_uri()
        return ddb.DBConnectionPool(
            host,
            port,
            threadNum,
            user,
            pwd,
            protocol=ddb.settings.PROTOCOL_DDB,
            show_output=True,
        )

    @property
    def session(self) -> ddb.Session:
        """直接获取会话对象"""
        return self._session

    @property
    def pool(self) -> ddb.DBConnectionPool:
        """直接获取连接池对象"""
        if self._pool_instance is None:
            self._pool_instance = self._create_pool()
        return self._pool_instance
    
    
    @classmethod
    def close_pool(cls):
        """关闭并清理连接池"""
        if cls._pool_instance is not None:
            with cls._pool_lock:
                if cls._pool_instance is not None:
                    try:
                        cls._pool_instance.shutDown()
                    except:
                        pass
                    cls._pool_instance = None
                    
    def tableAppender(self, db_path: str, table_name: str, data: pd.DataFrame) -> None:
        """
        将数据追加到指定的表中。

        :param db_path: 数据库路径
        :type db_path: str
        :param table_name: 表名
        :type table_name: str
        :param data: 要追加的数据
        :type data: pd.DataFrame
        :raises ValueError: 如果数据库路径或表名无效
        """
        session = self.get_session()

        if not session.existsDatabase(db_path):
            raise ValueError(f"{db_path} is not a valid database")

        if not session.existsTable(db_path, table_name):
            raise ValueError(f"{table_name} is not a valid table")

        # 获取表的列名
        table_cols: List[str] = (
            session.loadTable(table_name, db_path).schema["name"].to_list()
        )
        # 数据列名顺序与表列名顺序一致
        data: pd.DataFrame = data[table_cols]

        appender: ddb.tableAppender = ddb.tableAppender(
            tableName=table_name,
            ddbSession=session,
            dbPath=db_path,
        )

        appender.append(data)

    def tableUpsert(
        self,
        db_path: str,
        table_name: str,
        data: pd.DataFrame,
        keyColNames: Optional[Union[str, List]] = None,
        sortColumns: Optional[Union[str, List]] = None,
    ) -> None:
        """
        将数据帧中的数据插入或更新到指定的数据库表中。

        :param db_path: 数据库路径。
        :param table_name: 表名。
        :param data: 包含要插入或更新的数据的数据帧。
        :param keyColNames: 用于确定唯一记录的列名。默认为None。
        :param sortColumns: 用于排序的列名。默认为None。
        :return: 无返回值。
        """
        session = self.get_session()

        if not session.existsDatabase(db_path):
            raise ValueError(f"{db_path} is not a valid database")

        if not session.existsTable(db_path, table_name):
            raise ValueError(f"{table_name} is not a valid table")

        # 获取表的列名
        table_cols: List[str] = (
            session.loadTable(table_name, db_path).schema["name"].to_list()
        )
     
        # 数据列名顺序与表列名顺序一致
        data: pd.DataFrame = data[table_cols]

        if keyColNames is None:
            keyColNames: List = []

        if sortColumns is None:
            sortColumns: List = []

        upserter: ddb.tableUpsert = ddb.tableUpsert(
            tableName=table_name,
            ddbSession=session,
            dbPath=db_path,
            keyColNames=keyColNames,
            sortColumns=sortColumns,
        )
        upserter.upsert(data)
    
    # def __del__(self):
    #     """析构函数，确保资源正确释放"""
    #     try:
    #         if hasattr(self, '_session') and self._session:
    #             self._session.close()
    #     except:
    #         pass

if __name__ == "__main__":

    uri: str = "dolphindb://admin:123456@114.80.110.170:28848"

    # 测试python端安装ddb插件
    config = DDBConnectionSpec(uri=uri)
    connector = DDBClient(config)

    # expr:str = """
    # installPlugin("lgbm")
    # loadPlugin("lgbm")
    # """
    # connector.session.run(expr)
