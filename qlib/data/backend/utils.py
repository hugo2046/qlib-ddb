# qlib/data/backend/utils.py
"""
QLib Backend 层的通用工具函数

这个模块包含所有 backend 实现共享的工具函数：
- URI 处理和脱敏
- 连接管理
- Backend 间共享的辅助函数
"""

import re
from typing import Optional


def mask_uri(uri: Optional[str]) -> Optional[str]:
    """
    脱敏 URI 中的敏感信息（用户名和密码）

    Parameters
    ----------
    uri : str, optional
        数据库连接 URI，格式如: dolphindb://username:password@host:port

    Returns
    -------
    str, optional
        脱敏后的 URI，格式如: dolphindb://uxxxn:xxxxx@host:port
        如果输入为 None 或非字符串，原样返回

    Examples
    --------
    >>> mask_uri("dolphindb://admin:123456@172.17.0.1:8848")
    'dolphindb://axxn:xxxxx@172.17.0.1:8848'
    """
    if not uri or not isinstance(uri, str):
        return uri

    # 匹配 protocol://username:password@host:path 格式
    pattern = r'^(.*?://)([^:@]+):([^@]+)@([^/]+)(.*)$'
    match = re.match(pattern, uri)

    if match:
        protocol = match.group(1)
        username = match.group(2)

        # 用户名脱敏：保留首尾字符，中间用 x 填充
        if len(username) > 2:
            masked_username = f"{username[0]}{'x' * (len(username) - 2)}{username[-1]}"
        elif len(username) == 2:
            masked_username = f"{username[0]}x"
        else:
            masked_username = "x"

        # 密码完全脱敏为星号（固定 5 个星号）
        masked_password = "*****"

        return f"{protocol}{masked_username}:{masked_password}@{match.group(4)}{match.group(5)}"

    # 如果 URI 不匹配标准格式，原样返回
    return uri
