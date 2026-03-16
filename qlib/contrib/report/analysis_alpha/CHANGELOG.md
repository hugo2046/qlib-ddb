<!--
 * @Author: Hugo
 * @Date: 2026-01-30 13:46:51
 * @LastEditors: shen.lan123@gmail.com
 * @LastEditTime: 2026-01-30 13:50:50
 * @Description: 
-->
# FactorStats模块重构总结
本次任务将dzstat的
Stats
类核心功能成功整合到了Qlib的架构中，并进行了完全重写。

##主要变更
1. **架构升级**
   
保留了qlib的分层架构，将代码组织为四个核心类：

- FactorGrouper: 负责因子的分位数分层（Quantile Grouping）。
- VectorExecutor: 核心计算引擎，负责IC计算、收益计算、交易逻辑处理和风控过滤。
- PerformanceMetrics: 负责绩效指标统计（Sharpe, MaxDD等）和报告生成。
- FactorAnalyzer: 用户接口（Facade），一站式管理分析流程。
2. 功能增强 (继承自 dzstat)
- **IC计算**:
    - 支持 Pearson 和 Spearman 两种相关性计算方法。
    - 支持 ic_method 参数配置。
    - **PNL计算**:
    - 完整复刻了金融工程逻辑，包括持仓收益 (`ins_pnl`) 和交易收益 (`trd_pnl`) 的拆分。
    - 实现了基于市值的归一化处理。
  - **精细化过滤**:
    - 实现了基于涨跌停 (`limit_status`) 和停牌 (`suspend_status`) 的精细化仓位过滤逻辑。
  - **报告生成**:
    - 提供了按年/月等周期聚合的详细绩效报告，包含收益、IC、IR、Sharpe、最大回撤（含起止时间）等指标。
  
3. API改进

    新接口更加简洁且符合DataFrame处理习惯：

    ```python
    from qlib.contrib.report.analysis_alpha.factor_stats import FactorAnalyzer
    # 初始化
    analyzer = FactorAnalyzer(
        n_groups=5,
        ic_method='spearman',
        filter_limit=True
    )
    # 运行全量分析
    pnl, detail, perf = analyzer.run_analysis(
        factor, daily_ret, trade_ret, 
        limit_status=limit_status, 
        suspend_status=suspend_status
    )
    # 打印报告
    analyzer.metrics.generate_report(perf)
    # 运行分层回测
    q_results = analyzer.calc_quantile(...)
    ```

**验证结果**
通过 test_factor_stats.py脚本进行了完整验证：

1. IC计算: 能够正确计算并输出IC序列。
2. PNL计算: 多空组合收益、换手率等计算正确。
3. 分层回测: Group 1 到 Group 5 分层逻辑正常工作。
4. 报告格式: 控制台输出格式清晰，包含所有关键指标。
测试脚本运行结果显示所有功能正常，无报警信息。

## 2026-01-30 修复：数据对齐与鲁棒性增强

针对外部数据接入时常见的索引对齐问题（IC全为NaN/Ret全为0），进行了以下增强：

1.  **自动时区处理 (TZ-Naive Enforcement)**:
    *   新增 `_ensure_tz_naive` 内部方法。
    *   在 `__init__` 和 `run_analysis` 入口处，强制移除所有输入 DataFrame (`factor`, `daily_ret`, `settle_price` 等) 的时区信息。
    *   解决了 `QlibDataLoader` 加载的数据（通常带 UTC 时区）与本地构建的因子（通常无时区）无法即时对齐的问题。

2.  **数据维度检查**:
    *   增加了对 `factor` 索引类型的检查。
    *   如果传入的 DataFrame 索引不是 `DatetimeIndex`（例如因 `unstack` 方向错误导致的 Stock 为 Index），现在会输出明确的 Warning 提示用户检查数据形状。

3.  **create_factor_analyzer_from_dataframes 增强**:
    *   在数据适配层同样增加了时区移除逻辑，确保通过该接口创建的 Analyzer 实例内部数据完全对齐。
    *   增强了类型检查的报错信息。