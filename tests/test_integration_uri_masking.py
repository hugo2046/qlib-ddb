# tests/test_integration_uri_masking.py
"""测试 URI 脱敏在 qlib.init 中的集成"""

import sys
import logging
from io import StringIO
import pytest

# 确保能导入 qlib
sys.path.insert(0, "/data1/hugo/workspace/qlib_ddb")


def test_init_logs_masked_uri(caplog):
    """测试 qlib.init 日志输出不包含明文密码"""
    # 注意：这个测试需要实际的 DolphinDB 服务器，或者需要 mock
    # 如果没有测试环境，可以跳过

    import qlib
    from qlib.config import REG_CN

    # 使用 caplog 捕获日志
    with caplog.at_level(logging.INFO):
        try:
            qlib.init(
                database_uri="dolphindb://admin:123456@172.17.0.1:8848",
                region=REG_CN,
            )
        except Exception as e:
            # 连接失败是预期的，我们只检查日志
            pass

    # 检查所有日志记录
    for record in caplog.records:
        log_message = record.getMessage()

        # 如果日志包含 data_path，验证密码已脱敏
        if "data_path" in log_message:
            assert "admin:123456" not in log_message, f"发现明文密码在日志中: {log_message}"
            assert "admin:***" in log_message, f"密码未正确脱敏: {log_message}"
            print(f"✓ 日志安全: {log_message}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
