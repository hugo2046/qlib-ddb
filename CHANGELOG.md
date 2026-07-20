# CHANGELOG

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
