"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2025-02-19 14:29:41
Description: 用于连接ddb
"""

import threading

import dolphindb as ddb
from pydantic import BaseModel, field_validator
from urllib.parse import unquote, urlparse

from ....log import get_module_logger

logger = get_module_logger("ddb_client")


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
    """DolphinDB 连接客户端：持有一个会话与一个惰性创建的连接池。

    - 会话在构造时建立（reconnect=True 提供断线重连）。
    - 连接池为实例属性且惰性创建：目标服务器为社区版（2 核/8GB）时，
      每条连接都占用服务器内存，只有真正用到时才创建。

    :param config: 连接配置（含 dolphindb:// URI）
    """

    def __init__(self, config: DDBConnectionSpec):
        self._config = config
        self._session = self._create_session()
        # ⚠️ 连接池必须是实例属性：类变量会导致连接不同服务器的多个客户端共享同一个池
        self._pool_instance: ddb.DBConnectionPool | None = None
        self._pool_lock = threading.Lock()

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

    def _create_pool(self, threadNum: int = 4) -> ddb.DBConnectionPool:
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
        """惰性获取连接池对象（线程安全的双重检查）"""
        if self._pool_instance is None:
            with self._pool_lock:
                if self._pool_instance is None:
                    self._pool_instance = self._create_pool()
        return self._pool_instance

    def close_pool(self) -> None:
        """关闭并清理连接池（未创建过则为空操作）"""
        with self._pool_lock:
            if self._pool_instance is not None:
                try:
                    self._pool_instance.shutDown()
                except Exception as e:
                    logger.warning(f"关闭连接池失败: {e}")
                self._pool_instance = None

    def close(self) -> None:
        """显式关闭会话与连接池。

        不提供 __del__：本客户端的会话会被 provider/storage 等多处共享，
        GC 时机关闭共享会话会破坏其他使用方（见 DolphinDBDataLoader 历史 bug）。
        """
        self.close_pool()
        try:
            self._session.close()
        except Exception as e:
            logger.warning(f"关闭会话失败: {e}")
