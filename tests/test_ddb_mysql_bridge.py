"""DDBMySQLBridge 的离线单元测试（不依赖 DolphinDB / MySQL 服务器）。

历史 bug 回归：``load_mysql_plugin`` 曾误装 "lgbm" 插件而非 "mysql"，
导致后续 ``mysql::connect`` 必然失败。
"""

import pandas as pd

from qlib.data.backend.ddb_qlib.ddb_mysql_bridge import DDBMySQLBridge


class _FakeSession:
    """记录 run 脚本的假会话。"""

    def __init__(self, raise_on_run: bool = False):
        self.scripts: list[str] = []
        self.raise_on_run = raise_on_run

    def run(self, script: str):
        if self.raise_on_run:
            raise RuntimeError("boom")
        self.scripts.append(script)
        return pd.DataFrame()


def _make_bridge(session: _FakeSession) -> DDBMySQLBridge:
    """绕过构造器（构造器会真实连接数据库）注入假会话。"""
    bridge = object.__new__(DDBMySQLBridge)
    bridge.ddb_session = session
    return bridge


def test_load_mysql_plugin_installs_mysql_not_lgbm():
    session = _FakeSession()
    bridge = _make_bridge(session)
    bridge.load_mysql_plugin()
    script = "\n".join(session.scripts)
    assert 'installPlugin("mysql")' in script
    assert 'loadPlugin("mysql")' in script
    assert "lgbm" not in script, "回归：曾误装 lgbm 插件"


def test_close_swallows_error():
    """close 失败不应中断程序执行。"""
    bridge = _make_bridge(_FakeSession(raise_on_run=True))
    bridge.close()  # 不应抛出
