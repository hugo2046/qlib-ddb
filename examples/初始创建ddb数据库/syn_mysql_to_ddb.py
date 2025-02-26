"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2025-02-20 15:03:54
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2025-02-26 17:13:40
Description: 同步数据库
"""

import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, PROJECT_ROOT)


from qlib.data.backend.ddb_qlib.ddb_mysql_bridge import (
    DDBMySQLBridge,
    init_qlib_ddb_from_mysql,
)

import fire


ddb_uri: str = "dolphindb://admin:123456@hostlocal:28848"
mysql_uri: str = (
    "mysql+mysqlconnector://root:40678@hostlocal:5333/windDB"
)


def main(ddb_uri: str = ddb_uri, mysql_uri: str = mysql_uri):
    # 测试导入初始数据
    # NOTE:FEATRUE表重复运行会生成重复值。是直接添加的数据。
    init_qlib_ddb_from_mysql(ddb_uri, mysql_uri)


if __name__ == "__main__":

    # 测试查询show_tables
    # bridge: DDBMySQLBridge = DDBMySQLBridge(ddb_uri, mysql_uri)
    # print(bridge.show_tables())

    fire.Fire(main)
