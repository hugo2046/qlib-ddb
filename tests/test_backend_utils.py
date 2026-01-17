# tests/test_backend_utils.py
"""测试 Backend 工具函数"""

import pytest
from qlib.data.backend.utils import mask_uri


class TestMaskUri:
    """测试 URI 脱敏功能（遵循 SQLAlchemy 行业标准）"""

    def test_mask_dolphindb_uri_with_password(self):
        """测试 DolphinDB URI 脱敏"""
        uri = "dolphindb://admin:123456@172.17.0.1:8848"
        masked = mask_uri(uri)

        # 验证密码已脱敏（用户名保持可见，符合行业标准）
        assert "admin:123456" not in masked
        assert "admin:***" in masked

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

    def test_mask_username_visible(self):
        """测试用户名保持可见（行业标准）"""
        # 1 个字符
        uri = "dolphindb://a:pass@host:8848"
        assert mask_uri(uri) == "dolphindb://a:***@host:8848"

        # 2 个字符
        uri = "dolphindb://ab:pass@host:8848"
        assert mask_uri(uri) == "dolphindb://ab:***@host:8848"

        # 长用户名
        uri = "dolphindb://administrator:pass@host:8848"
        masked = mask_uri(uri)
        assert masked == "dolphindb://administrator:***@host:8848"

    def test_mask_mysql_uri(self):
        """测试 MySQL URI 脱敏"""
        uri = "mysql://root:password@localhost:3306/mydb"
        masked = mask_uri(uri)

        # 验证密码脱敏（用户名可见）
        assert "root:password" not in masked
        assert "root:***" in masked
        assert "mysql" in masked
        assert "localhost" in masked
        assert masked == "mysql://root:***@localhost:3306/mydb"

    def test_mask_uri_with_database_path(self):
        """测试带数据库路径的 URI"""
        uri = "postgresql://user:secret@localhost:5432/mydb?sslmode=require"
        masked = mask_uri(uri)
        # 验证密码脱敏
        assert "user:secret" not in masked
        assert "user:***" in masked
        assert "localhost:5432" in masked
        assert masked == "postgresql://user:***@localhost:5432/mydb?sslmode=require"

    def test_uri_without_password(self):
        """测试不包含密码的 URI（不应脱敏）"""
        uri = "dolphindb://admin@host:8848"
        assert mask_uri(uri) == "dolphindb://admin@host:8848"

    def test_invalid_uri_format(self):
        """测试完全无效的 URI"""
        assert mask_uri("not-a-uri") == "not-a-uri"
