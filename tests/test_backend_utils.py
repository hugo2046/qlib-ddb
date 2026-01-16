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
