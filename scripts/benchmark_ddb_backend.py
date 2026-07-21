"""DDB 后端 live 基准脚本（可选，需要真实 DolphinDB 服务器）。

用法::

    DDB_BENCH_URI="dolphindb://admin:123456@host:8848" python scripts/benchmark_ddb_backend.py

对 D.calendar / D.instruments / D.features（纯字段与计算表达式）计时，
并通过包装 session.run/upload 统计 RPC 往返数，用于优化前后对比。
不设 DDB_BENCH_URI 时直接退出（离线 CI 安全）。
"""

import functools
import os
import sys
import time
from pathlib import Path

# 保证从仓库根目录可直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger


def main() -> None:
    uri = os.environ.get("DDB_BENCH_URI")
    if not uri:
        logger.info("未设置 DDB_BENCH_URI，跳过 live 基准")
        return

    import qlib
    from qlib.config import REG_CN
    from qlib.data import D

    qlib.init(database_uri=uri, region=REG_CN)

    # 包装共享 session 统计 RPC
    from qlib.data.data import DBClient

    session = DBClient.session
    counts = {"run": 0, "upload": 0, "loadTable": 0}

    def _count(name, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            counts[name] += 1
            return func(*args, **kwargs)

        return wrapper

    session.run = _count("run", session.run)
    session.upload = _count("upload", session.upload)
    session.loadTable = _count("loadTable", session.loadTable)

    def bench(label, fn):
        for key in counts:
            counts[key] = 0
        t0 = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0
        size = len(result) if hasattr(result, "__len__") else "-"
        logger.info(
            f"{label}: {elapsed:.3f}s, rows={size}, "
            f"RPC(run={counts['run']}, upload={counts['upload']}, loadTable={counts['loadTable']})"
        )
        return result

    start, end = "2023-01-01", "2023-12-31"

    bench("D.calendar", lambda: D.calendar(start_time=start, end_time=end))
    instruments = bench(
        "D.instruments+list",
        lambda: D.list_instruments(D.instruments("csi300"), start_time=start, end_time=end, as_list=True),
    )
    sample = instruments[:50]

    bench(
        "D.features 纯字段×2",
        lambda: D.features(sample, ["$close", "$open"], start_time=start, end_time=end),
    )
    bench(
        "D.features 计算×2",
        lambda: D.features(
            sample, ["Ref($close,1)/$close-1", "Mean($volume,5)"], start_time=start, end_time=end
        ),
    )
    # 二次调用：观测缓存生效后的 RPC 下降
    bench(
        "D.features 计算×2 (二次)",
        lambda: D.features(
            sample, ["Ref($close,1)/$close-1", "Mean($volume,5)"], start_time=start, end_time=end
        ),
    )


if __name__ == "__main__":
    main()
