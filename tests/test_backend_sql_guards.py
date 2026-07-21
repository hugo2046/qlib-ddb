"""MySQL 同步 SQL 参数校验的离线单元测试。

背景：``QlibDDBMySQLInitializer`` 的同步方法把用户可控的日期/代码/交易所参数
直接 f-string 拼入 SQL（唯一真实的注入面）。修复方式为拼接前做白名单校验：
- 日期：``^\\d{8}$``（Wind YYYYMMDD 约定）
- 标识符/代码：``^[A-Za-z0-9._-]+$``（覆盖合法 Wind 代码如 000300.SH）
合法输入原样通过，行为不变；仅病态输入提前抛 ValueError。
"""

import pytest

from qlib.data.backend.ddb_qlib.utils import validate_date_str, validate_sql_identifier


class TestValidateDateStr:
    def test_valid_dates_pass_through(self):
        assert validate_date_str("20240101") == "20240101"
        assert validate_date_str("19991231") == "19991231"

    @pytest.mark.parametrize(
        "bad",
        ["2024-01-01", "2024.01.01", "202401", "20240101; DROP TABLE x", "", "20240101'"],
    )
    def test_invalid_dates_rejected(self, bad):
        with pytest.raises(ValueError):
            validate_date_str(bad)


class TestValidateSqlIdentifier:
    def test_legitimate_values_pass_through(self):
        # 合法 Wind 代码、交易所、库表名必须原样通过（行为不变）
        for value in ["000300.SH", "SSE", "SZSE", "ASHAREEODPRICES", "csi_300", "H30021.CSI"]:
            assert validate_sql_identifier(value) == value

    @pytest.mark.parametrize(
        "bad",
        ["'; DROP TABLE x--", "SSE'", "a b", 'x"y', "code;", "(select 1)", ""],
    )
    def test_injection_payloads_rejected(self, bad):
        with pytest.raises(ValueError):
            validate_sql_identifier(bad)


class TestInitializerGuards:
    """同步方法在构造 SQL 前即拒绝病态参数（不触达任何数据库连接）。"""

    def _make_initializer(self):
        from qlib.data.backend.ddb_qlib.ddb_mysql_bridge import QlibDDBMySQLInitializer

        init = object.__new__(QlibDDBMySQLInitializer)
        init._bridge = None  # 校验应在 _get_bridge 之前发生
        return init

    def test_sync_index_daily_rejects_bad_code(self):
        init = self._make_initializer()
        with pytest.raises(ValueError):
            init.sync_index_daily(["000300.SH'; DROP TABLE AINDEXEODPRICES--"])

    def test_sync_index_daily_rejects_bad_date(self):
        init = self._make_initializer()
        with pytest.raises(ValueError):
            init.sync_index_daily(["000300.SH"], start_date="2024-01-01")
