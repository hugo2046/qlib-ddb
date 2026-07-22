"""DolphinDB 会话的可复用记录型 mock（离线测试基础设施）。

模拟 ``fetch_features_from_ddb`` / storage 所需的最小 SDK 表面：
``run`` / ``upload`` / ``runFile`` / ``existsTable`` / ``loadTable``（含
``select/where/sort/exec/toDF/schema`` 链式调用），并对每类 RPC 计数，
供 RPC 往返数基线测试与分支回归测试共用。
"""

import re
from typing import Callable

import numpy as np
import pandas as pd


class FakeQueryChain:
    """模拟 loadTable 返回的表句柄及其链式查询。"""

    def __init__(self, session: "RecordingSession", table_name: str, db_path: str):
        self._session = session
        self.table_name = table_name
        self.db_path = db_path
        self.selects: list = []
        self.wheres: list[str] = []
        self.sorts: list = []
        self.exec_col: str | None = None

    # --- 链式查询 ---
    def select(self, cols):
        self.selects.append(cols)
        return self

    def where(self, cond: str):
        self.wheres.append(cond)
        return self

    def sort(self, cols):
        self.sorts.append(cols)
        return self

    def exec(self, col: str):
        self.exec_col = col
        return self

    def toDF(self):
        self._session.query_chains.append(self)
        return self._session._resolve_table_result(self)

    @property
    def schema(self) -> pd.DataFrame:
        cols = self._session.table_columns.get(
            (self.db_path, self.table_name),
            self._session.table_columns.get(self.table_name, []),
        )
        return pd.DataFrame({"name": cols})


class RecordingSession:
    """记录所有 RPC 调用的假 DolphinDB 会话。

    :param calendar: 交易日历（np.ndarray[datetime64]），供 TradeDateUtils 使用
    :param table_columns: {(db_path, table_name) 或 table_name: [列名]}，供 schema 查询
    :param table_results: {(db_path, table_name) 或 table_name: DataFrame 或 callable(chain)}，
        供 toDF 返回查询结果
    :param run_responses: [(正则, 结果或 callable(script))]，按序匹配 run 脚本
    """

    def __init__(
        self,
        calendar: np.ndarray | None = None,
        table_columns: dict | None = None,
        table_results: dict | None = None,
        run_responses: list[tuple[str, object]] | None = None,
    ):
        self.calendar = calendar
        self.table_columns = table_columns or {}
        self.table_results = table_results or {}
        self.run_responses = run_responses or []

        # RPC 计数与调用记录
        self.counts: dict[str, int] = {
            "run": 0,
            "upload": 0,
            "runFile": 0,
            "existsTable": 0,
            "loadTable": 0,
        }
        self.run_scripts: list[str] = []
        self.uploads: list[dict] = []
        self.run_files: list = []
        self.load_table_calls: list[tuple[str, str]] = []
        self.query_chains: list[FakeQueryChain] = []
        self.exists_result: bool = True  # existsTable 的返回值（可按测试配置）

    # --- SDK 表面 ---
    def run(self, script: str):
        self.counts["run"] += 1
        self.run_scripts.append(script)
        for pattern, result in self.run_responses:
            if re.search(pattern, script):
                return result(script) if isinstance(result, Callable) else result
        return None

    def upload(self, variables: dict):
        self.counts["upload"] += 1
        self.uploads.append(variables)

    def runFile(self, filepath):
        self.counts["runFile"] += 1
        self.run_files.append(filepath)

    def existsTable(self, db_path: str, table_name: str) -> bool:
        self.counts["existsTable"] += 1
        return self.exists_result

    def loadTable(self, tableName: str = None, dbPath: str = None, **kwargs):
        # 兼容位置参数写法 loadTable(table, db)
        table = tableName or kwargs.get("tableName")
        db = dbPath or kwargs.get("dbPath")
        self.counts["loadTable"] += 1
        self.load_table_calls.append((db, table))
        return FakeQueryChain(self, table, db)

    # --- 结果解析 ---
    def _resolve_table_result(self, chain: FakeQueryChain):
        # 日历查询：exec("TRADE_DAYS")
        if chain.exec_col == "TRADE_DAYS":
            if self.calendar is None:
                raise AssertionError("测试未配置 calendar，但发生了日历查询")
            return self.calendar
        result = self.table_results.get(
            (chain.db_path, chain.table_name),
            self.table_results.get(chain.table_name),
        )
        if callable(result):
            return result(chain)
        if result is None:
            raise AssertionError(
                f"测试未配置表结果: {chain.db_path}/{chain.table_name}"
            )
        return result


def make_calendar(start: str, periods: int) -> np.ndarray:
    """构造连续工作日的假交易日历（np.datetime64 数组，升序）。"""
    return pd.date_range(start, periods=periods, freq="B").values
