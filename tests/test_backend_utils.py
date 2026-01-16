# tests/test_backend_utils.py
"""测试 Backend 工具函数"""

import pytest
from qlib.data.backend.utils import mask_uri


class TestMaskUri:
    """测试 URI 脱敏功能"""

    def test_mask_dolphindb_uri_with_password(self):
        """测试 DolphinDB URI 脱敏"""
        uri = "dolphindb://admin:123456@172.17.0.1:8848"
        masked = mask_uri(uri)

        # 验证敏感信息已脱敏
        assert "admin" not in masked
        assert "123456" not in masked

        # 验证调试信息保留
        assert "dolphindb://" in masked
        assert "172.17.0.1" in masked
        assert "8848" in masked

    def test_invalid_uri_returns_none(self):
        """测试 None 输入"""
        assert mask_uri(None) is None

    def test_empty_string(self):
        """测试空字符串"""
        assert mask_uri("") == ""

    def test_mask_short_username(self):
        """测试短用户名脱敏"""
        # 1 个字符
        uri = "dolphindb://a:pass@host:8848"
        assert mask_uri(uri) == "dolphindb://x:*****@host:8848"

        # 2 个字符
        uri = "dolphindb://ab:pass@host:8848"
        assert mask_uri(uri) == "dolphindb://ax:*****@host:8848"

    def test_mask_long_username(self):
        """测试长用户名脱敏"""
        uri = "dolphindb://administrator:pass@host:8848"
        masked = mask_uri(uri)
        assert masked == "dolphindb://axxxxxxxxxxxr:*****@host:8848"

    def test_mask_mysql_uri(self):
        """测试 MySQL URI 脱敏"""
        uri = "mysql://root:password@localhost:3306/mydb"
        masked = mask_uri(uri)

        assert "root" not in masked
        assert "password" not in masked
        assert "mysql" in masked
        assert "localhost" in masked
        assert masked == "mysql://rxxt:*****@localhost:3306/mydb"

    def test_mask_uri_with_database_path(self):
        """测试带数据库路径的 URI"""
        uri = "postgresql://user:secret@localhost:5432/mydb?sslmode=require"
        masked = mask_uri(uri)
        assert "user" not in masked
        assert "secret" not in masked
        assert "localhost:5432" in masked
        assert masked == "postgresql://uxxr:*****@localhost:5432/mydb?sslmode=require"

    def test_uri_without_password(self):
        """测试不包含密码的 URI（不应脱敏）"""
        uri = "dolphindb://admin@host:8848"
        assert mask_uri(uri) == "dolphindb://admin@host:8848"

    def test_invalid_uri_format(self):
        """测试完全无效的 URI"""
        assert mask_uri("not-a-uri") == "not-a-uri"
