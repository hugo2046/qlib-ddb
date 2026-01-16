# tests/test_config_masking.py
"""测试 URI 脱敏配置选项"""

import sys
sys.path.insert(0, "/data1/hugo/workspace/qlib_ddb")

import qlib
from qlib.config import REG_CN


def test_masking_enabled_by_default():
    """测试默认启用脱敏"""
    from qlib.data.backend.utils import mask_uri

    uri = "dolphindb://admin:123456@host:8848"
    masked = mask_uri(uri)

    assert "admin:123456" not in masked
    assert "axxxn:*****" in masked


def test_can_disable_masking_via_config():
    """测试可以通过配置禁用脱敏"""
    # 这个测试需要完整的 qlib.init 上下文
    # 可能需要 mock 或者实际的配置加载
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
