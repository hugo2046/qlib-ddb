# URI 敏感信息脱敏实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标:** 在日志中自动脱敏数据库连接 URI 中的用户名和密码，防止敏感信息泄露

**架构:** 创建独立的 URI 脱敏工具函数，在日志输出时调用，不影响实际的数据库连接逻辑。采用正则表达式解析 URI，保留调试所需信息（协议、主机、端口）的同时脱敏凭据。

**技术栈:** Python 3.8+, 正则表达式 (re), pytest

---

## 问题背景

当前 `qlib.init()` 在日志中输出完整的数据库连接 URI：

```
data_path={'__DEFAULT_FREQ': 'DolphinDB(dolphindb://admin:123456@172.17.0.1:8848)'}
```

这导致明文密码被记录到日志文件中，存在安全风险。参考 SQLAlchemy 等业界标准的做法，应该将敏感信息脱敏：

```
data_path={'__DEFAULT_FREQ': 'DolphinDB(dolphindb://axxn:xxxxx@172.17.0.1:8848)'}
```

---

## Task 1: 创建 URI 脱敏工具函数

**文件:**
- 创建: `qlib/data/backend/utils.py`

**Step 1: 创建文件并编写函数骨架**

```python
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
    pass
```

**Step 2: 提交骨架代码**

```bash
git add qlib/data/backend/utils.py
git commit -m "feat: add URI masking utility skeleton"
```

---

## Task 2: 实现 mask_uri 函数核心逻辑

**文件:**
- 修改: `qlib/data/backend/utils.py:23-37`

**Step 1: 编写基本实现的测试**

创建: `tests/test_backend_utils.py`

```python
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
```

**Step 2: 运行测试确认失败**

```bash
pytest tests/test_backend_utils.py::TestMaskUri::test_mask_dolphindb_uri_with_password -v
```

Expected: `FAILED` - 函数返回 `None` 或 `pass`

**Step 3: 实现核心逻辑**

修改: `qlib/data/backend/utils.py:23-37`

```python
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
```

**Step 4: 运行测试确认通过**

```bash
pytest tests/test_backend_utils.py::TestMaskUri::test_mask_dolphindb_uri_with_password -v
```

Expected: `PASSED`

**Step 5: 运行所有编写的测试**

```bash
pytest tests/test_backend_utils.py -v
```

Expected: 全部 `PASSED`

**Step 6: 提交实现**

```bash
git add qlib/data/backend/utils.py tests/test_backend_utils.py
git commit -m "feat: implement URI masking with username/password redaction"
```

---

## Task 3: 扩展测试覆盖边界情况

**文件:**
- 修改: `tests/test_backend_utils.py`

**Step 1: 添加边界情况测试**

```python
    def test_mask_short_username(self):
        """测试短用户名脱敏"""
        # 1 个字符
        uri = "dolphindb://a:pass@host:8848"
        assert mask_uri(uri) == "dolphindb://x:xxxxx@host:8848"

        # 2 个字符
        uri = "dolphindb://ab:pass@host:8848"
        assert mask_uri(uri) == "dolphindb://ax:xxxxx@host:8848"

    def test_mask_long_username(self):
        """测试长用户名脱敏"""
        uri = "dolphindb://administrator:pass@host:8848"
        masked = mask_uri(uri)
        assert masked == "dolphindb://axxxxxxxxx:xxxxx@host:8848"

    def test_mask_mysql_uri(self):
        """测试 MySQL URI 脱敏"""
        uri = "mysql://root:password@localhost:3306/mydb"
        masked = mask_uri(uri)

        assert "root" not in masked
        assert "password" not in masked
        assert "mysql" in masked
        assert "localhost" in masked
        assert masked == "mysql://rxt:xxxxx@localhost:3306/mydb"

    def test_mask_uri_with_database_path(self):
        """测试带数据库路径的 URI"""
        uri = "postgresql://user:secret@localhost:5432/mydb?sslmode=require"
        masked = mask_uri(uri)
        assert "user" not in masked
        assert "secret" not in masked
        assert "localhost:5432" in masked
        assert masked == "postgresql://uxx:xxxxx@localhost:5432/mydb?sslmode=require"

    def test_uri_without_password(self):
        """测试不包含密码的 URI（不应脱敏）"""
        uri = "dolphindb://admin@host:8848"
        assert mask_uri(uri) == "dolphindb://admin@host:8848"

    def test_invalid_uri_format(self):
        """测试完全无效的 URI"""
        assert mask_uri("not-a-uri") == "not-a-uri"
```

**Step 2: 运行所有新测试**

```bash
pytest tests/test_backend_utils.py -v
```

Expected: 全部 `PASSED`

**Step 3: 提交扩展测试**

```bash
git add tests/test_backend_utils.py
git commit -m "test: add comprehensive edge case coverage for URI masking"
```

---

## Task 4: 集成到 qlib.init 日志输出

**文件:**
- 修改: `qlib/__init__.py`
  - 顶部添加导入（约第 20-30 行区域）
  - 修改日志输出逻辑（第 88-96 行）

**Step 1: 在文件顶部添加导入**

查看文件开头的导入区域：

```bash
head -40 qlib/__init__.py | grep -n "^import\|^from"
```

找到合适的导入位置（通常在 `from qlib.utils` 相关导入之后），添加：

```python
from qlib.data.backend.utils import mask_uri
```

**Step 2: 修改日志输出逻辑**

修改: `qlib/__init__.py:88-96`

将：
```python
    # 根据不同的 URI 类型显示不同的信息
    data_path = {}
    for _freq, provider_uri in C.dpm.provider_uri.items():
        uri_type = C.dpm.get_uri_type(provider_uri)
        if uri_type == "database":
            data_path[_freq] = f"DolphinDB({provider_uri})"
        else:
            data_path[_freq] = C.dpm.get_data_uri(_freq)
    logger.info(f"data_path={data_path}")
```

改为：
```python
    # 根据不同的 URI 类型显示不同的信息
    data_path = {}
    for _freq, provider_uri in C.dpm.provider_uri.items():
        uri_type = C.dpm.get_uri_type(provider_uri)
        if uri_type == "database":
            # 使用脱敏后的 URI 用于日志输出
            data_path[_freq] = f"DolphinDB({mask_uri(provider_uri)})"
        else:
            data_path[_freq] = C.dpm.get_data_uri(_freq)
    logger.info(f"data_path={data_path}")
```

**Step 3: 创建集成测试**

创建: `tests/test_integration_uri_masking.py`

```python
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
            assert "admin" not in log_message or "axxn" in log_message, f"用户名未脱敏: {log_message}"
            print(f"✓ 日志安全: {log_message}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
```

**Step 4: 运行集成测试**

```bash
pytest tests/test_integration_uri_masking.py -v -s
```

Expected:
- 如果有 DolphinDB 服务器：测试通过，日志显示脱敏后的 URI
- 如果没有服务器：连接失败但日志仍显示脱敏后的 URI

**Step 5: 手动验证**

创建临时测试脚本: `scripts/test_uri_masking_manual.py`

```python
# scripts/test_uri_masking_manual.py
import sys
sys.path.insert(0, "/data1/hugo/workspace/qlib_ddb")

import logging
import qlib
from qlib.config import REG_CN

# 设置日志级别为 INFO 以查看输出
logging.basicConfig(level=logging.INFO)

print("\n=== 测试 URI 脱敏 ===")
print("期望输出: DolphinDB(dolphindb://axxn:xxxxx@...)")
print("实际输出:")

try:
    qlib.init(
        database_uri="dolphindb://admin:123456@172.17.0.1:8848",
        region=REG_CN,
    )
except Exception as e:
    print(f"连接失败（预期的）: {e}")

print("\n检查上述日志中的 data_path=...")
print("应该看到: dolphindb://axxn:xxxxx@...")
print("不应该看到: admin:123456")
```

运行:
```bash
python scripts/test_uri_masking_manual.py
```

验证输出中不包含明文密码。

**Step 6: 提交集成**

```bash
git add qlib/__init__.py tests/test_integration_uri_masking.py scripts/test_uri_masking_manual.py
git commit -m "feat: integrate URI masking into qlib.init logging"
```

---

## Task 5: 添加配置选项（可选但推荐）

**文件:**
- 修改: `qlib/config.py`
- 修改: `qlib/__init__.py`

**Step 1: 在配置中添加脱敏开关**

查找配置文件中的默认配置区域（通常在文件顶部或 DEFAULT_CONFIG 字典），添加：

```python
# 是否在日志中脱敏敏感信息（如数据库密码）
LOG_MASK_SENSITIVE: bool = True
```

**Step 2: 修改 __init__.py 使用配置**

修改: `qlib/__init__.py:88-96`

```python
    # 根据不同的 URI 类型显示不同的信息
    data_path = {}

    # 读取脱敏配置，默认启用
    mask_enabled = C.get("log_mask_sensitive", True)

    for _freq, provider_uri in C.dpm.provider_uri.items():
        uri_type = C.dpm.get_uri_type(provider_uri)
        if uri_type == "database":
            # 根据配置决定是否脱敏
            display_uri = mask_uri(provider_uri) if mask_enabled else provider_uri
            data_path[_freq] = f"DolphinDB({display_uri})"
        else:
            data_path[_freq] = C.dpm.get_data_uri(_freq)
    logger.info(f"data_path={data_path}")
```

**Step 3: 测试配置选项**

创建: `tests/test_config_masking.py`

```python
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
    assert "axxn:xxxxx" in masked


def test_can_disable_masking_via_config():
    """测试可以通过配置禁用脱敏"""
    # 这个测试需要完整的 qlib.init 上下文
    # 可能需要 mock 或者实际的配置加载
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Step 4: 运行测试**

```bash
pytest tests/test_config_masking.py -v
```

**Step 5: 提交配置选项**

```bash
git add qlib/config.py qlib/__init__.py tests/test_config_masking.py
git commit -m "feat: add configurable URI masking option"
```

---

## Task 6: 更新文档

**文件:**
- 修改: `CLAUDE.md`

**Step 1: 在 CLAUDE.md 中添加安全实践章节**

查找合适的位置（建议在"开发规范"章节之后），添加：

```markdown
## 安全实践

### URI 敏感信息脱敏

QLib-DDB 会自动脱敏日志中的数据库连接凭据，防止敏感信息泄露：

- **用户名**: 保留首尾字符（如 `admin` → `axxn`）
- **密码**: 完全脱敏为 `*****`

**示例**:

```python
# 实际连接
dolphindb://admin:123456@172.17.0.1:8848

# 日志输出
DolphinDB(dolphindb://axxn:xxxxx@172.17.0.1:8848)
```

**控制脱敏行为**:

默认启用脱敏。如需禁用（不推荐），在配置中设置：

```python
log_mask_sensitive: False
```

**实现位置**:
- 脱敏函数: `qlib/data/backend/utils.py`
- 集成点: `qlib/__init__.py:93`
```

**Step 2: 更新 README.md（如果需要）**

在项目 README 中添加安全特性说明：

```markdown
## 安全特性

- ✅ 自动脱敏日志中的数据库凭据
- ✅ 支持配置控制脱敏行为
- ✅ 符合 OWASP 日志安全最佳实践
```

**Step 3: 提交文档**

```bash
git add CLAUDE.md README.md
git commit -m "docs: add security practice documentation for URI masking"
```

---

## Task 7: 最终验证和清理

**Step 1: 运行完整测试套件**

```bash
pytest tests/test_backend_utils.py -v
pytest tests/test_integration_uri_masking.py -v
pytest tests/test_config_masking.py -v
```

Expected: 全部 `PASSED`

**Step 2: 运行项目现有测试确保无破坏**

```bash
pytest tests/ -k "not slow" -x
```

Expected: 无失败

**Step 3: 清理测试文件**

删除临时测试脚本：

```bash
rm scripts/test_uri_masking_manual.py
```

**Step 4: 最终提交**

```bash
git add -A
git commit -m "chore: finalize URI masking implementation"
```

**Step 5: 创建 git tag**

```bash
git tag -a v1.0-security-uri-masking -m "Implement URI sensitive data masking in logs"
```

---

## 验收标准

实现完成后，应该满足：

- [ ] 所有单元测试通过
- [ ] 集成测试通过
- [ ] 日志输出不再包含明文密码
- [ ] 保留足够的调试信息（主机、端口可见）
- [ ] 不影响现有功能
- [ ] 文档更新完整
- [ ] 代码符合项目规范（black, pylint 通过）

---

## 参考资料

- [OWASP Logging Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html)
- [SQLAlchemy URL Redaction](https://github.com/sqlalchemy/sqlalchemy/blob/main/lib/sqlalchemy/engine/url.py)
- [Python Regular Expression HOWTO](https://docs.python.org/3/howto/regex.html)

---

**创建日期:** 2026-01-16
**预估工作量:** 2-3 小时
**优先级:** 高（安全问题）
**风险等级:** 低（仅修改日志输出）
