# CHANGELOG

## 2026-07-21

### DDB 后端系统性优化（分支 optimize/ddb-backend，17 个原子提交）

面向 DolphinDB **社区版（2 核 / 8GB）** 的一轮优化：省往返、省传输、
省服务器内存、控会话数；公共接口 `D.features()/D.calendar()/D.instruments()`
的签名与返回结果完全不变。全部改动配套离线单测（mock 会话，共 115 项通过）。

**Bug 修复**：

- `ddb_dataset_processor` 在传入 `inst_processors` 时提前 `return pd.DataFrame()`，
  并行处理结果被整体丢弃（`data.py:782`）。
- `DDBClient` 连接池：`_pool_instance` 类变量导致多客户端共享同一池；
  `close_pool` 引用不存在的 `_pool_lock` 且永远读到 None（从未真正关闭过池）；
  删除调用不存在 `get_session()` 的死方法 `tableAppender/tableUpsert` 与
  `__main__` 中的硬编码凭据。
- `DolphinDBClientProvider` init 时急切创建 4 连接的池（无消费方）→ 改懒创建。
- `load_mysql_plugin` 误装 `lgbm` 插件（应为 `mysql`）。
- `DolphinDBDataLoader.__exit__/__del__` 无条件关闭进程级共享 session →
  引入 `_owns_session` 所有权标志。

**并发/线程安全**：

- 新增会话级 `DBClient.session_lock`（RLock）：feature 查询是
  「run→upload→run」多步会话对话，交错执行会互相覆盖服务器变量
  （跨线程数据污染根因）；所有 session 触点统一持锁。
- `QlibDataLoader` 全局 `_load_lock` 收窄至 DDB 路径，文件后端恢复无锁并行。

**健壮性/质量**：

- 12 处 `print` → loguru；3 处裸 `except:` 收敛；异常重包装补 `from e`。
- MySQL 同步 SQL 参数白名单校验（`validate_date_str`/`validate_sql_identifier`，
  唯一真实注入面）。
- 清理死代码（注释块 ×2、零调用者函数 ×2、16KB 备份脚本）与
  `OPERATOR_MAPPING` 重复键（生效映射不变，ast 测试锁定）。

**性能**（RPC 往返：计算分支每批 3-4 次 → 预热后 2 次）：

- 交易日历模块级缓存：Alpha158 一次 `D.features` 从 ~6 次全量日历下载 → 0；
  `DBCalendarStorage.index()/__getitem__` 改走 `H["c"]` 缓存。
- 日期字面量内联（消灭独立上传往返）；纯字段分支合并为单条 SQL 脚本；
  上传字典按分支裁剪。`fetch_features_from_ddb` 拆为编排器 + 4 个 helper。
- existsTable 仅正向缓存、股票池走 `H["i"]`、表 schema 进程内缓存、
  表达式翻译 lru_cache；统一由 `ddb_qlib.invalidate_ddb_caches()` 失效
  （写路径自动调用）。
- 计算分支结果直构 `(instrument, datetime)` MultiIndex 面板，替代
  concat→unstack→stack→swaplevel 的 ~4 次全景拷贝（保留 legacy 兜底 +
  逐值等价测试）。
- 三个 alpha 因子库（约 119KB）按字段前缀惰性加载（`Cannot recognize the
  token` 兜底重试；`C["ddb_preload_alpha_libs"]` 逃生开关）。
- 写路径按 URI 复用共享 `DDBClient`（批量导入 N 会话 → 1）。
- 批次参数配置化：`C["ddb_field_chunk_size"]=30`、`C["ddb_days_step"]=252`
  （默认值不变，`scripts/benchmark_ddb_backend.py` 供 live 校准）。

**明确不做**（2 核/8GB 约束）：读路径不引入 `DBConnectionPool` 并发、
不调高 mr 并行度、不做服务器端全景 pivot、不引入 module/functionView 持久化。


## 2026-07-20

### fix: DDB 批量取数路径恢复成分股 spans（入池/出池）过滤 ⚠️ BREAKING 行为变更

**问题**：`D.features(D.instruments("csi300"), ...)` 在 DolphinDB 模式下返回历史上
所有曾入池股票的全区间行情——出池不截断、调离不剔除。spans（入池/出池日期区间）
在 `DBFeatureStorage.__init__` 被 `list(map(...))` 迭代 dict 键时静默丢弃。

**修复**：

- `qlib/data/storage/dolphindb_storage.py`：`DBFeatureStorage.__init__` 对 dict
  入参保持原型（仅键做 `.upper()`），使 `fetch_features_from_ddb` 走其本就支持的
  dict 分支（`createDateStockMapping` + `conditionalFilter` 服务端过滤）。
- `qlib/data/backend/ddb_qlib/ddb_features.py`：非纯字段分支（带算子表达式，走
  `FeatureEngineeringByDate`）经 `mr` 分布式执行时 worker 端无法还原
  `conditionalFilter` 所需的 dict（报 "filterMap must be a dictionary"），故该分支
  传键列表全量计算，新增 `apply_spans_mask` 在 Python 侧对结果补 spans 掩码——
  先算后掩码的语义与原版 qlib 文件后端 `inst_calculator` 一致。

**⚠️ BREAKING**：所有用市场字符串（"csi300"/"csi500"/"ashares"）取数的下游，
股票池口径从「历史全体成员全历史」收窄为「point-in-time 动态成分股」：
截面股票数明显变少（如 csi300 2024 年从 ~896 只收窄为 ~336 只），出池日后数据
截断。这是预期中的正确行为变化，但下游回测结果会随之改变。

**回归测试**：`tests/test_ddb_storage_spans.py`（离线单测，不依赖 DDB 服务器）。
