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


class TestMaskUriPublicHost:
    """测试公网主机脱敏（仅公网地址遮蔽，内网/本机保持可见）"""

    def test_mask_public_ipv4(self):
        """公网 IPv4 部分遮蔽（保留首尾段），端口保留"""
        uri = "dolphindb://admin:123456@114.80.110.170:28848"
        assert mask_uri(uri) == "dolphindb://admin:***@114.**.**.170:28848"

    def test_private_ipv4_kept(self):
        """内网 IP 保持可见"""
        assert mask_uri("dolphindb://admin:pass@172.17.0.1:8848") == "dolphindb://admin:***@172.17.0.1:8848"
        assert mask_uri("dolphindb://admin:pass@192.168.1.10:8848") == "dolphindb://admin:***@192.168.1.10:8848"
        assert mask_uri("dolphindb://admin:pass@10.0.0.5:8848") == "dolphindb://admin:***@10.0.0.5:8848"

    def test_loopback_and_localhost_kept(self):
        """回环地址和 localhost 保持可见"""
        assert mask_uri("dolphindb://admin:pass@127.0.0.1:8848") == "dolphindb://admin:***@127.0.0.1:8848"
        assert mask_uri("dolphindb://admin:pass@localhost:8848") == "dolphindb://admin:***@localhost:8848"

    def test_mask_public_ip_without_credentials(self):
        """无凭据 URI 中的公网 IP 同样遮蔽"""
        uri = "dolphindb://114.80.110.170:8848"
        assert mask_uri(uri) == "dolphindb://114.**.**.170:8848"

    def test_mask_domain_host(self):
        """多标签域名遮蔽中段"""
        uri = "dolphindb://admin:pass@ddb.mycorp.com:8848"
        assert mask_uri(uri) == "dolphindb://admin:***@ddb.**.com:8848"

    def test_mask_public_ip_with_path(self):
        """带路径/参数的 URI 主机遮蔽后其余部分保留"""
        uri = "mysql://root:secret@114.80.110.170:3306/mydb?sslmode=require"
        assert mask_uri(uri) == "mysql://root:***@114.**.**.170:3306/mydb?sslmode=require"

    def test_mask_public_ipv6_bracketed(self):
        """带方括号的公网 IPv6 整体遮蔽，端口保留"""
        uri = "dolphindb://admin:pass@[2400:da00::6666]:8848"
        assert mask_uri(uri) == "dolphindb://admin:***@[***]:8848"

    def test_loopback_ipv6_bracketed_kept(self):
        """带方括号的回环 IPv6 保持可见"""
        uri = "dolphindb://admin:pass@[::1]:8848"
        assert mask_uri(uri) == "dolphindb://admin:***@[::1]:8848"

    def test_mask_public_ipv6_bare(self):
        """裸公网 IPv6（无方括号、无端口）同样整体遮蔽"""
        uri = "dolphindb://admin:pass@2400:da00::6666"
        assert mask_uri(uri) == "dolphindb://admin:***@***"

    def test_mask_public_ip_username_without_password(self):
        """有用户名无密码时公网 IP 仍遮蔽"""
        uri = "dolphindb://admin@114.80.110.170:8848"
        assert mask_uri(uri) == "dolphindb://admin@114.**.**.170:8848"

    def test_mask_two_label_domain(self):
        """2 标签域名保留前 2 字符遮蔽"""
        uri = "dolphindb://admin:pass@mycorp.com:8848"
        assert mask_uri(uri) == "dolphindb://admin:***@my***.com:8848"
