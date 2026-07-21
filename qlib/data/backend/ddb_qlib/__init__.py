"""
Author: hugo2046 shen.lan123@gmail.com
Date: 2025-02-20 23:52:37
LastEditors: hugo2046 shen.lan123@gmail.com
LastEditTime: 2025-04-23 10:40:54
Description:
"""

from .ddb_client import DDBConnectionSpec, DDBClient
from .ddb_operator import (
    DDBTableOperator,
    create_calendar_table,
    create_feature_daily_table,
    create_instrument_table,
    clean_qlib_db,
    write_df_to_ddb
)
from .ddb_mysql_bridge import DDBMySQLBridge, init_qlib_ddb_from_mysql
from .schemas import QlibTableSchema
from .ddb_features import (
    register_ddb_functions_to_qlib,
    fetch_features_from_ddb,
    normalize_fields_to_ddb,
    TradeDateUtils,
    adapt_qlib_expr_syntax_for_ddb,
)


def invalidate_ddb_caches() -> None:
    """清空 DDB 后端的进程内缓存。

    覆盖范围：
    - :class:`TradeDateUtils` 的模块级交易日历缓存；
    - ``H["c"]`` 中 DBCalendarStorage 的原始日历缓存。

    写路径（``write_df_to_ddb`` / CSV 导入 / ``clean_qlib_db``）变更表数据后
    会自动调用；长驻只读进程若感知到外部写入，也可手动调用本函数。
    """
    TradeDateUtils.clear_cache()
    try:
        from qlib.data.cache import H

        cal_cache = H["c"]
        while len(cal_cache):
            cal_cache.popitem(last=False)
    except Exception:  # pragma: no cover - 缓存清理失败不应阻断写路径
        pass
