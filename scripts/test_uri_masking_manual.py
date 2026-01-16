# scripts/test_uri_masking_manual.py
import sys
sys.path.insert(0, "/data1/hugo/workspace/qlib_ddb")

import logging
import qlib
from qlib.config import REG_CN

# 设置日志级别为 INFO 以查看输出
logging.basicConfig(level=logging.INFO)

print("\n=== 测试 URI 脱敏 ===")
print("期望输出: DolphinDB(dolphindb://axxn:*****@...)")
print("实际输出:")

try:
    qlib.init(
        database_uri="dolphindb://admin:123456@172.17.0.1:8848",
        region=REG_CN,
    )
except Exception as e:
    print(f"连接失败（预期的）: {e}")

print("\n检查上述日志中的 data_path=...")
print("应该看到: dolphindb://axxxn:*****@...")
print("不应该看到: admin:123456")
