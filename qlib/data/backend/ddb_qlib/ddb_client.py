'''
Author: hugo2046 shen.lan123@gmail.com
Date: 2025-02-19 14:29:41
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2025-02-25 12:30:03
Description: 用于连接ddb
'''
from urllib.parse import unquote, urlparse

import dolphindb as ddb
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
            enableASYNC=False
        )
        session.connect(host=host, port=port, userid=user, password=pwd, reconnect=True)
        return session

    @property
    def session(self) -> ddb.Session:
        """直接获取会话对象"""
        return self._session


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