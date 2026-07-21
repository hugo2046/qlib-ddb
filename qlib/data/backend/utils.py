# qlib/data/backend/utils.py
"""
QLib Backend 层的通用工具函数

这个模块包含所有 backend 实现共享的工具函数：
- URI 处理和脱敏
- 连接管理
- Backend 间共享的辅助函数
"""

import ipaddress
import re
from typing import Optional


def _mask_host(host: str) -> str:
    """
    脱敏主机地址（仅遮蔽公网地址，内网/本机保持可见）

    规则：
    - 公网 IPv4：部分遮蔽，保留首尾段（如 ``114.80.110.170`` → ``114.**.**.170``）
    - 公网 IPv6：整体遮蔽为 ``***``
    - 内网 IP / 回环地址 / localhost / 单标签主机名：保持可见（便于调试）
    - 3 标签及以上域名：遮蔽中段（如 ``ddb.mycorp.com`` → ``ddb.**.com``）
    - 2 标签域名：保留前 2 字符（如 ``mycorp.com`` → ``my***.com``，
      只遮中段等于不遮，故改为遮蔽首标签主体）

    Parameters
    ----------
    host : str
        主机地址（IP 或域名，不含端口）

    Returns
    -------
    str
        脱敏后的主机地址
    """
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None

    if ip is not None:
        # 内网、回环、链路本地等非公网地址保持可见
        if not ip.is_global:
            return host
        if ip.version == 4:
            parts = host.split(".")
            return f"{parts[0]}.**.**.{parts[3]}"
        return "***"

    # 非 IP 主机名：单标签（localhost、内网主机名）保留，多标签域名遮蔽中段
    labels = host.split(".")
    if len(labels) == 1:
        return host
    if len(labels) == 2:
        return f"{labels[0][:2]}***.{labels[-1]}"
    return f"{labels[0]}.**.{labels[-1]}"


def _mask_hostport(hostport: str) -> str:
    """
    脱敏 ``host[:port]`` 片段，端口保持可见

    Parameters
    ----------
    hostport : str
        主机端口片段，如 ``114.80.110.170:28848``、``[::1]:8848`` 或 ``localhost``

    Returns
    -------
    str
        脱敏后的片段
    """
    # 整体先按 IP 尝试解析：覆盖无端口的 IPv4 及裸 IPv6（不带方括号时
    # 不能用 rpartition 拆分，否则末段会被误当端口导致公网 IPv6 漏遮）
    try:
        ipaddress.ip_address(hostport)
        return _mask_host(hostport)
    except ValueError:
        pass

    # IPv6 字面量带方括号：[addr]:port
    if hostport.startswith("["):
        end = hostport.find("]")
        if end != -1:
            return f"[{_mask_host(hostport[1:end])}]{hostport[end + 1:]}"

    host, sep, port = hostport.rpartition(":")
    if sep and port.isdigit():
        return f"{_mask_host(host)}:{port}"
    return _mask_host(hostport)


def mask_uri(uri: Optional[str]) -> Optional[str]:
    """
    脱敏 URI 中的敏感信息（密码与公网主机地址）

    脱敏规则：
    - 密码：完全脱敏为 ``***``（遵循 SQLAlchemy/psycopg2 行业标准）
    - 用户名：保持可见（便于调试，通常不是敏感信息）
    - 主机：仅公网地址遮蔽（公网 IPv4 保留首尾段），内网 IP、
      localhost 等保持可见；多标签域名遮蔽中段
    - 端口：保持可见（便于区分多实例）

    Parameters
    ----------
    uri : str, optional
        数据库连接 URI，格式如: dolphindb://username:password@host:port

    Returns
    -------
    str, optional
        脱敏后的 URI，格式如: dolphindb://username:***@host:port
        如果输入为 None 或非字符串，原样返回

    Examples
    --------
    >>> mask_uri("dolphindb://admin:123456@172.17.0.1:8848")
    'dolphindb://admin:***@172.17.0.1:8848'
    >>> mask_uri("dolphindb://admin:123456@114.80.110.170:28848")
    'dolphindb://admin:***@114.**.**.170:28848'
    """
    if not uri or not isinstance(uri, str):
        return uri

    # 匹配 protocol://[username[:password]@]host[:port][/path] 格式
    # 密码允许包含 `:` 和 `/`（宁可对罕见的"无凭据但 query 含 @"的 URI
    # 误判，也不能让含特殊字符的真实密码因不匹配而全文泄露）
    pattern = r"^(?P<protocol>.*?://)(?:(?P<username>[^:@/]+)(?::(?P<password>[^@]+))?@)?(?P<hostport>[^/?#]+)(?P<rest>.*)$"
    match = re.match(pattern, uri)

    if not match:
        # 如果 URI 不匹配标准格式，原样返回
        return uri

    protocol = match.group("protocol")
    username = match.group("username")
    password = match.group("password")
    hostport = _mask_hostport(match.group("hostport"))
    rest = match.group("rest")

    if username is None:
        return f"{protocol}{hostport}{rest}"
    if password is None:
        return f"{protocol}{username}@{hostport}{rest}"
    # 密码完全脱敏为星号（遵循 SQLAlchemy 标准：使用 3 个星号）
    return f"{protocol}{username}:***@{hostport}{rest}"
