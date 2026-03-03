import pandas as pd
import numpy as np
from loguru import logger
from typing import Optional, Dict, Union, Tuple

try:
    from rich.console import Console
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# ==========================================
# 1. 异常与辅助类
# ==========================================
class DataQualityError(Exception):
    """自定义数据质量异常"""

    pass


# ==========================================
# 0. 工具函数 (Utilities)
# ==========================================
def pretty_table_print(df: pd.DataFrame, title: str = "Table"):
    """
    使用 Rich 库打印美观的表格

    :param df: 数据 DataFrame
    :type df: pd.DataFrame
    :param title: 表格标题, defaults to "Table"
    :type title: str
    """
    if not RICH_AVAILABLE:
        # 如果 rich 不可用，回退使用普通的 logger 打印表格数据
        logger.info(f"\n{title}\n{df.to_string()}")
        return

    console = Console(width=160)
    table = Table(title=title)

    # 添加列定义的展示
    # 索引列
    table.add_column(
        df.index.name or "Index", justify="left", style="cyan", no_wrap=True
    )

    # 添加数据列名
    for col in df.columns:
        table.add_column(str(col), justify="right")

    # 填充表格行数据
    for idx, row in df.iterrows():
        # 处理索引列的格式化展示
        idx_str = str(idx)
        # 对数值内容进行格式化处理
        vals = []
        for v in row:
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))

        table.add_row(idx_str, *vals)

    # 尽量使用 loguru 进行日志输出。由于 rich.Console 默认直接输出到 stdout，
    # 我们在此捕获其产生的输出字符串，随后将其交由 loguru 记录，以保持输出一致性。

    with console.capture() as capture:
        console.print(table)
    str_output = capture.get()
    logger.info(f"\n{str_output}")


# ==========================================
# 2. # 2. 因子分层 (数据处理层)
# ==========================================
class FactorGrouper:
    """
    因子分层处理器 (因子分位数分组器)

    职责：负责将因子值转化为分层持仓权重 (Portfolios)。

    :param n_groups: 分组数量, 默认为 5
    :param ascending: 是否升序排列 (True: 小值在 Group1; False: 大值在 Group1), 默认为 False
    :param ignore_zeros: 是否忽略 0 值, 默认为 False
    :param max_nan_rate: 最大允许的 NaN 比例, 默认为 0.9
    :param strict_mode: 是否开启严格模式 (报错而非警告), 默认为 False
    """

    def __init__(
        self,
        n_groups: int = 5,
        ascending: bool = True,  # 默认升序，Group 1为最小值
        ignore_zeros: bool = False,
        max_nan_rate: float = 0.9,
        strict_mode: bool = False,
    ):
        self.n_groups = n_groups
        self.ascending = ascending
        self.ignore_zeros = ignore_zeros
        self.max_nan_rate = max_nan_rate
        self.strict_mode = strict_mode

    def get_quantile_groups(self, factor: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        生成分层持仓组合 (持仓权重)
        优化版实现：边界处理与原有stat逻辑保持一致。

        :param factor: 因子DataFrame (index=date, columns=stock)
        :type factor: pd.DataFrame
        :return: 分层后的持仓权重字典 {group_id: weights_df}
        :rtype: Dict[int, pd.DataFrame]
        """
        work_factor = factor.copy()
        if self.ignore_zeros:
            work_factor = work_factor.replace(0, np.nan)

        # 数据质量检查（获取初始统计信息）
        initial_stats = self._check_data_quality(work_factor)

        # 计算分位数边界
        quantiles = np.linspace(0, 1, self.n_groups + 1)
        # q_vals: 索引为日期，列为分位数序列
        # 使用插值方法 'linear' 或根据需求调整，pandas 默认使用 linear
        q_vals = work_factor.quantile(quantiles, axis=1).T

        # 调整列名以便后续处理 (1, 2, ..., n_groups+1)
        # 我们的目标是生成 1..n_groups 的分组
        # q_vals的列原本是 0.0, 0.2, ... 1.0
        # 我们用序列 0 到 n_groups 来标识边界
        q_columns = range(self.n_groups + 1)
        q_vals.columns = q_columns

        # 原有stat逻辑：第一组下界为最小值再减去一个常数，兼容后续的半开区间筛选逻辑
        # 注意：如果 ascending=False (默认), 则数值越大排名越靠前。
        # 这里我们先按数值分层，最后根据ascending参数决定label的顺序

        # 为了兼容半开区间 (low, high]，我们需要处理下界
        q_vals[0] = q_vals[0] - 1e-4

        portfolios = {}
        for i in range(self.n_groups):
            # 分组 i+1 : (q_vals[i], q_vals[i+1]]
            # 最后一组包含上界
            if i == self.n_groups - 1:
                mask = work_factor.gt(q_vals[i], axis=0)
            else:
                mask = work_factor.gt(q_vals[i], axis=0) & work_factor.le(
                    q_vals[i + 1], axis=0
                )

            # 执行组内等权分配处理
            weights = mask.astype(float).div(mask.sum(axis=1), axis=0)

            # 确定分组ID
            # 如果 ascending=True (升序)，数值越小Group ID越小 -> i=0 (小值) -> Group 1
            # 如果 ascending=False (降序)，数值越大Group ID越小 -> i=n-1 (大值) -> Group 1
            # 但通常习惯是：Group 1代表"最好"的一组，或者按数值大小排列
            # 这里的实现：Group 1 到 Group N 对应 数值小 到 数值大
            # 如果用户希望 Group 1 是最大值组，需要在外部 interpret，或者我们在这里反转 key

            group_id = i + 1
            if not self.ascending:
                # 如果是降序排列（因子越大越好），则数值最大的组为 Group 1
                group_id = self.n_groups - i

            portfolios[group_id] = weights

        # 生成数据丢失报告（参考 Alphalens）
        self._report_data_loss(initial_stats, portfolios, max_loss=self.max_nan_rate)

        return portfolios

    def _check_data_quality(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        检查因子数据质量并返回初始统计信息

        参考 Alphalens 的数据质量检查逻辑，适配展开式 DataFrame 数据结构。

        :param df: 因子 DataFrame (index=date, columns=stock)
        :type df: pd.DataFrame
        :return: 包含初始统计信息的字典
        :rtype: Dict[str, float]
        """
        # 1. 初始数据统计
        initial_total = float(df.size)  # 总元素数（日期数 × 股票数）
        initial_valid = float(df.count().sum())  # 有效值总数
        initial_nan_count = initial_total - initial_valid
        initial_nan_rate = (
            initial_nan_count / initial_total if initial_total > 0 else 0.0
        )

        # 2. 每日数据统计
        daily_valid_count = df.count(axis=1)  # 每日有效股票数
        daily_total_count = df.shape[1]  # 每日总股票数

        # 3. 分组能力检查
        # 检查每日是否有足够的不同值来进行分组
        # 理论上需要至少 n_groups 个不同的有效值
        daily_unique_count = df.apply(lambda row: row.dropna().nunique(), axis=1)
        insufficient_days = (daily_unique_count < self.n_groups).sum()
        total_days = len(df.index)
        insufficient_rate = insufficient_days / total_days if total_days > 0 else 0.0

        # 4. 高 NaN 率检查
        daily_nan_rates = 1.0 - (daily_valid_count / daily_total_count)
        max_daily_nan_rate = daily_nan_rates.max() if len(daily_nan_rates) > 0 else 0.0
        high_nan_days = (daily_nan_rates > self.max_nan_rate).sum()
        high_nan_rate = high_nan_days / total_days if total_days > 0 else 0.0

        # 5. 生成警告信息
        warnings = []

        if initial_nan_rate > 0.5:
            warnings.append(f"初始因子数据中有 {initial_nan_rate:.1%} 的缺失值")

        if insufficient_days > 0:
            warnings.append(
                f"有 {insufficient_days}/{total_days} ({insufficient_rate:.1%}) 个交易日的不同有效值数量少于分组数 {self.n_groups}，"
                f"这些日期的分组结果可能不理想"
            )

        if high_nan_days > 0:
            warnings.append(
                f"有 {high_nan_days}/{total_days} ({high_nan_rate:.1%}) 个交易日的 NaN 比例超过阈值 {self.max_nan_rate:.1%}"
            )

        # 6. 输出警告或报错
        if warnings:
            warning_msg = "数据质量警告:\n" + "\n".join(f"  - {w}" for w in warnings)
            if self.strict_mode and (insufficient_rate > 0.1 or high_nan_rate > 0.1):
                raise DataQualityError(warning_msg)
            else:
                logger.warning(warning_msg)

        # 7. 返回统计信息供后续使用
        return {
            "initial_total": initial_total,
            "initial_valid": initial_valid,
            "initial_nan_count": initial_nan_count,
            "initial_nan_rate": initial_nan_rate,
            "total_days": float(total_days),
            "insufficient_days": float(insufficient_days),
            "insufficient_rate": insufficient_rate,
            "max_daily_nan_rate": max_daily_nan_rate,
        }

    def _report_data_loss(
        self,
        initial_stats: Dict[str, float],
        portfolios: Dict[int, pd.DataFrame],
        max_loss: float = 0.35,
    ):
        """
        报告数据丢失情况，参考 Alphalens 的报告格式

        :param initial_stats: 初始数据统计信息（来自 _check_data_quality）
        :type initial_stats: Dict[str, float]
        :param portfolios: 分组后的持仓权重字典
        :type portfolios: Dict[int, pd.DataFrame]
        :param max_loss: 最大允许的数据丢失率阈值, 默认为 0.35
        :type max_loss: float
        """
        # 1. 计算分组后的有效数据量
        # 注意：weights 是归一化后的权重，非零权重代表有效分配
        grouped_valid = 0.0
        for weights_df in portfolios.values():
            # 统计非零权重的数量（即有效分配到该组的股票-日期对数量）
            grouped_valid += float((weights_df > 0).sum().sum())

        # 2. 计算各阶段数据丢失
        initial_total = initial_stats["initial_total"]
        initial_valid = initial_stats["initial_valid"]
        initial_nan_count = initial_stats["initial_nan_count"]

        # 因子缺失导致的丢失
        factor_loss_count = initial_nan_count
        factor_loss_rate = (
            factor_loss_count / initial_total if initial_total > 0 else 0.0
        )

        # 分组过程导致的丢失（边界处理、无法分组等）
        # 注意：由于我们使用等权重分配，理论上所有有效值都应该被分配
        # 但实际可能因为分位数边界重叠等原因导致部分数据未分配
        binning_loss_count = initial_valid - grouped_valid
        binning_loss_rate = (
            binning_loss_count / initial_total if initial_total > 0 else 0.0
        )

        # 总丢失率
        total_loss_count = initial_total - grouped_valid
        total_loss_rate = total_loss_count / initial_total if initial_total > 0 else 0.0

        # 3. 生成报告
        report_lines = [
            "=" * 70,
            "数据质量检查报告 (参考 Alphalens 格式)",
            "=" * 70,
            f"初始因子数据: {int(initial_total)} 个数据点",
            f"  - 有效值: {int(initial_valid)} ({initial_valid/initial_total:.1%})",
            f"  - 缺失值 (NaN): {int(initial_nan_count)} ({factor_loss_rate:.1%})",
            f"",
            f"分组后有效数据: {int(grouped_valid)} 个权重分配",
            f"",
            f"数据丢失统计:",
            f"  - 因子缺失导致: {int(factor_loss_count)} ({factor_loss_rate:.1%})",
            f"  - 分组过程导致: {int(binning_loss_count)} ({binning_loss_rate:.1%})",
            f"  - 总丢失率: {int(total_loss_count)} ({total_loss_rate:.1%})",
        ]

        # 4. 检查是否超过阈值
        if total_loss_rate > max_loss:
            report_lines.append("")
            report_lines.append(
                f"⚠️  警告: 总数据丢失率 {total_loss_rate:.1%} 超过阈值 {max_loss:.1%}"
            )
            if self.strict_mode:
                report_lines.append("=" * 70)
                error_msg = "\n".join(report_lines)
                raise DataQualityError(
                    f"数据丢失率超过阈值!\n{error_msg}\n"
                    f"建议: 增加 max_loss 参数或检查数据质量"
                )
            else:
                report_lines.append(f"   建议: 考虑增加 max_loss 参数或检查数据质量")
        else:
            report_lines.append("")
            report_lines.append(f"✓ 数据丢失率在可接受范围内 (阈值: {max_loss:.1%})")

        report_lines.append("=" * 70)

        # 5. 输出报告
        logger.info("\n" + "\n".join(report_lines))


# ==========================================
# 3. # 3. 向量化计算引擎 (计算层)
# ==========================================
class VectorExecutor:
    """
    向量化收益计算引擎 (向量化回测执行器)

    职责：
    1. 计算 IC (信息系数)
    2. 计算 PNL (拆分持仓收益与交易损耗收益)
    3. 执行精细化的涨跌停与停牌过滤逻辑

    :param booksize: 目标市值 (0 表示不固定市值), 默认为 0.0
    :param cost: 交易费率, 默认为 0.0
    :param filter_limit: 是否过滤涨跌停, 默认为 False
    :param filter_suspend: 是否过滤停牌, 默认为 False
    :param ic_method: IC 计算方法 ('pearson'/'spearman'), 默认为 "spearman"
    :param delay: 因子生效延迟 (天), 默认为 0
    """

    def __init__(
        self,
        booksize: float = 0.0,  # 0表示不固定市值
        cost: float = 0.0,
        filter_limit: bool = False,
        filter_suspend: bool = False,
        ic_method: str = "spearman",
        delay: int = 0,
    ):
        self.booksize = booksize
        self.cost = cost
        self.filter_limit = filter_limit
        self.filter_suspend = filter_suspend
        self.ic_method = ic_method
        self.delay = delay

    def calculate_information_coefficient(
        self, factor: pd.DataFrame, returns: pd.DataFrame, method: str = None
    ) -> pd.Series:
        """
        计算 IC (信息系数)
        完全对齐原始 factor_test.py 的 ic_calc 方法逻辑

        :param factor: 因子值 DataFrame
        :type factor: pd.DataFrame
        :param returns: 收益率 DataFrame (通常是交易复权收益率 trd_adjreturns)
        :type returns: pd.DataFrame
        :param method: 相关性计算方法 ('pearson' 或 'spearman'), 若为 None 则使用实例默认值
        :type method: str, 可选
        :return: IC序列 (index=date)
        :rtype: pd.Series
        """
        method = method or self.ic_method

        # ========================================
        # 复刻原始 factor_test.py 的 ic_calc 逻辑
        # ========================================

        # 1. 因子值截面排序（将因子值转换为百分位排名）
        # 原有stat逻辑：position_rk = position.rank(axis=1, method="average", pct=True)
        position_rk = factor.rank(axis=1, method="average", pct=True)

        # 2. 对齐收益率时间序列，严格按照原始逻辑处理delay参数
        # 原有stat关键逻辑: stock_return = self.trd_adjreturns.shift(self.delay - 1)
        # 这里需要特别注意：
        # - delay=0时，shift(-1)，使用t+1日收益（正常的预测逻辑）
        # - delay=1时，shift(0)，使用t日收益（可能因为数据时间戳定义特殊）
        stock_return = returns.shift(self.delay - 1)

        # 3. 对齐索引维度（提升后续筛选操作效率）
        # 原有stat：确保因子和收益率的索引和列完全对齐
        stock_return = stock_return.reindex(
            index=position_rk.index, columns=position_rk.columns
        )

        # 4. 仅保留仓位非空位置的日度收益并进行排序
        # 原有stat逻辑：这对多空仓位单边的IC计算至关重要
        # 只对有因子值的股票计算收益率排名
        return_rk = stock_return[position_rk.notna()].rank(
            axis=1, method="average", pct=True
        )

        # 5. 计算截面Pearson/Spearman相关系数
        # 原有stat：ic = position_rk.corrwith(return_rk, axis=1)
        ic = position_rk.corrwith(return_rk, axis=1, method=method)

        return ic

    def calculate_portfolio_returns(
        self,
        target_weights: pd.DataFrame,
        daily_ret: pd.DataFrame,
        trade_ret: pd.DataFrame = None,
        limit_status: Optional[pd.DataFrame] = None,
        suspend_status: Optional[pd.DataFrame] = None,
        return_detail: bool = False,
        position_type: int = 1,  # 1: 多头, -1: 空头
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
        """
        执行向量化回测
        完全复刻原始 factor_test.py 的 pnl_calc 方法逻辑

        :param target_weights: 目标持仓权重/因子值
        :type target_weights: pd.DataFrame
        :param daily_ret: 日收益率 (复权结算价收益率)
        :type daily_ret: pd.DataFrame
        :param trade_ret: 交易收益率差异 (可选), 默认为 None
        :type trade_ret: pd.DataFrame
        :param limit_status: 涨跌停状态 (1: 涨停, -1: 跌停, 0: 正常), 默认为 None
        :type limit_status: Optional[pd.DataFrame]
        :param suspend_status: 停牌状态 (1: 停牌, 0: 正常), 默认为 None
        :type suspend_status: Optional[pd.DataFrame]
        :param return_detail: 是否返回详细 PNL 信息, 默认为 False
        :type return_detail: bool
        :param position_type: 仓位类型 (1: 多头, -1: 空头), 默认为 1
        :type position_type: int
        :return: 回测结果 DataFrame, 或 (pnl_df, pnl_detail) 元组
        :rtype: Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]
        """

        # ========================================
        # 完全复刻原始 factor_test.py pnl_calc 的核心逻辑
        # ========================================

        # 1. 基础数据处理（原有stat第479-505行）
        # 注意：在原始 factor_test.py 中，delay 在主流程 calc() 中处理：
        # position = position.shift(delay) (第898行)
        # 这意味着 pnl_calc 接收的 position 已经是 delay 偏移后的
        position = target_weights.copy()

        # 为了保持与原始逻辑一致，我们在这里也对 position 做 delay 处理
        if self.delay != 0:
            position = position.shift(self.delay)

        position_adj_num = 0.000001  # 防止除零的调整值
        position_sum = position.abs().sum(axis=1) + position_adj_num  # 因子值绝对值累加
        daily_value = position_sum

        # 2. 市值归一化处理（原有stat第501-509行）
        if self.booksize > 0:
            # 固定市值模式：每日市值标准化到booksize
            book_unit = position_sum.rtruediv(self.booksize)
            position_factor = position.multiply(book_unit, axis="index")
            daily_value = pd.Series(self.booksize, index=position.index)
        else:
            # 因子值直接代表市值（非固定市值模式）
            position_factor = position.copy()
            daily_value = position_sum

        position_sum_before_filter = position_factor.abs().sum(axis=1)  # 过滤前的总仓位

        # ========================================
        # 开始过滤逻辑（原有stat第511-574行）
        # ========================================

        position_factor.fillna(0, inplace=True)  # 调用本方法前在过滤多空仓位时会留下nan

        # 3. 过滤涨跌停（原有stat第516-542行）
        if self.filter_limit and limit_status is not None:
            # 对齐涨跌停状态数据
            up_down_limit_status = limit_status.reindex(
                index=position_factor.index, columns=position_factor.columns
            ).fillna(0)

            # 按日期分组前向填充，确保日内一致性原有stat逻辑）
            up_down_limit_status = (
                up_down_limit_status.sort_index()
                .groupby(up_down_limit_status.index.date)
                .ffill()
            )

            # 涨跌停方向判断（原有stat逻辑）
            up_limit = up_down_limit_status == 1 * position_type  # 同向涨停
            down_limit = up_down_limit_status == -1 * position_type  # 同向跌停

            # 仓位变化计算（原有stat逻辑）
            position_change = position_factor.abs().diff()
            position_up = position_change > 0  # 加仓
            position_down = position_change < 0  # 减仓

            # 应用过滤规则：涨停不能同向加仓，跌停不能逆向减仓
            filter_mask = (position_up & up_limit) | (position_down & down_limit)
            position_factor[filter_mask] = np.nan

        # 4. 过滤停牌（原有stat第545-554行）
        if self.filter_suspend and suspend_status is not None:
            # 对齐停牌状态数据
            suspension_status = suspend_status.reindex(
                index=position_factor.index, columns=position_factor.columns
            ).fillna(0)

            # 按日期分组前向填充（原有stat逻辑）
            suspension_status = (
                suspension_status.sort_index()
                .groupby(suspension_status.index.date)
                .ffill()
            )

            # 停牌股票不能交易
            _filter = suspension_status == 1
            position_factor[_filter] = np.nan

        # 5. 仓位回填与重平衡（原有stat第556-574行）
        position_factor.ffill(inplace=True)  # NaN 股票保持前值
        position_factor.fillna(0, inplace=True)  # 前值均为 NaN 时用 0 填充

        position_sum_after_filter = position_factor.abs().sum(axis=1) + position_adj_num
        scale_final = position_sum_before_filter.div(
            position_sum_after_filter
        )  # 过滤前后仓位的比值
        position_factor = position_factor.mul(scale_final, axis=0)  # 重新缩放仓位

        # ========================================
        # 结束过滤逻辑
        # ========================================

        # 6. PNL 计算（原有stat第576-674行）
        daily_count = (position_factor != 0).sum(axis=1)  # 每日持仓个数
        last_position = position_factor.shift().fillna(
            0
        )  # 因子值向后偏移1位，使t-1日因子值与t日对齐

        # 持仓部分PNL：持仓市值相比前一日变化值
        daily_ins_pnl = last_position.multiply(daily_ret)

        # 交易部分计算（原有stat第582-614行）
        if self.booksize > 0:
            # 因子值不表示仓位的情况（原有stat优化算法1）
            daily_ins_trd = position_factor.diff()
        else:
            # 因子值表示仓位的情况（原有stat优化算法2）
            # 为保证无需交易的股票不产生收益，在做diff时将前一天仓位乘以前一天收益率
            daily_ins_trd = position_factor.sub(
                last_position.multiply(1 + daily_ret.shift().fillna(0))
            )

        # 修正起始日的日内交易额为持仓总额（原有stat逻辑）
        daily_ins_trd.iloc[0] = position_factor.iloc[0]

        # 交易部分PNL（原有stat第617-625行）
        if trade_ret is not None:
            # 交易部分的t日变化值（原有stat注释逻辑）
            daily_trd_pnl = -daily_ins_trd.multiply(trade_ret)
        else:
            # 如果没有提供交易收益率，则交易PNL为0
            daily_trd_pnl = pd.DataFrame(
                0, index=daily_ins_trd.index, columns=daily_ins_trd.columns
            )

        # 总PNL为持有部分pnl和交易部分pnl相加（原有stat第627-630行）
        daily_total_pnl = daily_ins_pnl.fillna(0) + daily_trd_pnl.fillna(0)

        # 每日收益累加（原有stat第632行）
        daily_pnl = daily_total_pnl[daily_total_pnl.abs() > 0].sum(axis=1, min_count=1)

        # 每日交易值累加（原有stat第635行）
        daily_trd = daily_ins_trd.abs().sum(axis=1)

        # 每日换手率计算（原有stat第637-642行）
        # 分母使用前一天总市值，如前一天市值为0，则使用当天市值
        last_daily_value = (
            daily_value.shift()
            .replace(position_adj_num, np.nan)
            .fillna(method="bfill", limit=1)
        )
        daily_tvr = daily_trd.truediv(last_daily_value)

        # 每日原始回报率计算（原有stat第644行）
        daily_rawret = daily_pnl.truediv(last_daily_value)

        # 回报扣减手续费（原有stat第646-649行）
        if self.cost > 0:
            daily_returns = (daily_pnl - daily_trd * self.cost).truediv(
                last_daily_value
            )
        else:
            daily_returns = daily_rawret

        # 7. 记录各项指标（原有stat第660-673行）
        # 为避免nan对后续聚合计算结果造成影响，此处统一fillna(0)
        pnl_df = pd.DataFrame(
            {
                "ret": daily_returns.fillna(0),
                "trd": daily_trd.fillna(0),
                "tvr": daily_tvr.fillna(0),
                "value": daily_value.fillna(0),
                "rawret": daily_rawret.fillna(0),
                "count": daily_count.fillna(0),
                "pnl": daily_pnl.fillna(0),
                "trd_pnl": daily_trd_pnl.fillna(0).sum(axis=1),
                "ins_pnl": daily_ins_pnl.fillna(0).sum(axis=1),
            }
        )

        pnl_detail = {
            "daily_ins_pnl": daily_ins_pnl,
            "daily_trd_pnl": daily_trd_pnl,
            "daily_total_pnl": daily_total_pnl,
            "daily_ins_trd": daily_ins_trd,
            "position_factor": position_factor,  # 返回最终仓位用于调试
        }

        if return_detail:
            return pnl_df, pnl_detail
        return pnl_df
        # 该部分已经在上面的替换中完成了


# ==========================================
# # 4. 绩效指标统计计算 (指标统计层)
# ==========================================
class PerformanceMetrics:
    """
    绩效指标统计详细逻辑 (Metrics)
    支持按周期聚合，生成报告。

    :param periods_per_year: 年化周期数, defaults to 250
    """

    def __init__(self, periods_per_year: int = 250):
        self.ann_factor = periods_per_year

    def classify_period(self, idx: pd.DatetimeIndex, report_type: str) -> pd.Series:
        """生成周期分组标签"""
        if report_type == "daily":
            return idx.year * 10000 + idx.month * 100 + idx.day
        elif report_type == "weekly":
            # year * 100 + week
            return idx.year * 100 + idx.isocalendar().week
        elif report_type == "monthly":
            return idx.year * 100 + idx.month
        elif report_type == "yearly":
            return idx.year
        else:
            raise ValueError(f"Unknown report type: {report_type}")

    def calculate_by_period(
        self,
        result_df: pd.DataFrame,
        ic: Optional[pd.Series] = None,
        benchmark_ret: Optional[pd.Series] = None,
        report_type: str = "yearly",
    ) -> pd.DataFrame:
        """
        按周期聚合计算绩效指标

        :param result_df: 回测结果 DataFrame (需包含 'ret', 'trd', 'value', 'pnl' 等列)
        :type result_df: pd.DataFrame
        :param ic: IC 序列, 默认为 None
        :type ic: Optional[pd.Series]
        :param benchmark_ret: 基准收益率 (暂未深度集成), 默认为 None
        :type benchmark_ret: Optional[pd.Series]
        :param report_type: 聚合周期类型 ('daily', 'weekly', 'monthly', 'yearly'), 默认为 "yearly"
        :type report_type: str
        :return: 聚合后的绩效指标 DataFrame
        :rtype: pd.DataFrame
        """
        df = result_df.copy()
        if ic is not None:
            df["ic"] = ic

        # 周期分组
        period_idx = self.classify_period(df.index, report_type)

        grouped = df.groupby(period_idx)

        # 聚合计算
        perf = pd.DataFrame()

        # 1. 收益统计 (Ret)
        perf["ret_mean"] = grouped["ret"].mean()
        perf["ret_std"] = grouped["ret"].std()
        perf["ret_sum"] = grouped["ret"].sum()

        # 2. 换手率 (Turnover)
        # 用总交易量 / 总市值
        sum_trd = grouped["trd"].sum()
        sum_val = grouped["value"].sum()
        perf["turnover"] = sum_trd / sum_val.replace(0, np.nan)

        # 3. IC统计
        if "ic" in df.columns:
            perf["ic_mean"] = grouped["ic"].mean()
            perf["ic_std"] = grouped["ic"].std()
            perf["ir"] = perf["ic_mean"] / perf["ic_std"].replace(0, np.nan)

        # 4. Sharpe (年化)
        # 这里的sharpe是周期内的sharpe，实际上我们通常输出年化sharpe
        # period sharpe = mean / std
        perf["sharpe"] = perf["ret_mean"] / perf["ret_std"].replace(0, np.nan)

        # 5. 最大回撤 (MaxDD)
        # 需要对每个 group 运行 max_dd
        dd_info = grouped["ret"].apply(self._calc_max_dd)
        perf["max_dd"] = dd_info.apply(lambda x: x[0])
        perf["max_dd_start"] = dd_info.apply(lambda x: x[1])
        perf["max_dd_end"] = dd_info.apply(lambda x: x[2])

        # 胜率
        perf["win_rate"] = grouped["ret"].apply(lambda x: (x > 0).mean())

        return perf

    def _calc_max_dd(self, ser: pd.Series):
        """
        计算最大回撤及起止时间
        完全对齐原始 factor_test.py 的 max_dd 方法逻辑

        :param ser: 收益率时间序列
        :type ser: pd.Series
        :return: [最大回撤绝对值, 开始时间, 结束时间]
        :rtype: List
        """

        # ========================================
        # 复刻原始 factor_test.py 的 max_dd 方法逻辑
        # ========================================

        # 初始化默认返回信息（原有stat逻辑）
        max_dd_info = [
            0,
            ser.index[0] if len(ser) > 0 else None,
            ser.index[-1] if len(ser) > 0 else None,
        ]

        # 检查数据有效性（原有stat逻辑）
        if len(ser) == 0 or ser[ser != 0].count() == 0:
            return max_dd_info

        # 如果一天交易多次，因暂时没有获取准确分时benchmark行情的方式，只能在天维度聚合后再统计
        # 计算累计收益序列 (1 + ret).cumprod()
        cum_ret = (1 + ser).cumprod()

        # 计算滚动最大值 expanding().max()
        rolling_max = cum_ret.expanding().max()

        # 计算回撤序列 (cum_ret / rolling_max) - 1
        drawdown = cum_ret / rolling_max - 1

        # 找到最大回撤（最小值）
        max_dd = drawdown.min()

        if pd.isna(max_dd) or max_dd >= 0:
            return max_dd_info

        # 找到最大回撤结束时间（最低点）
        max_dd_end = drawdown.idxmin()

        # 找到最大回撤开始时间（最低点之前的最高点）
        # 在最低点之前找最后一次达到peak的时间
        peak_value = rolling_max.loc[max_dd_end]
        mask_before_end = cum_ret.index <= max_dd_end
        candidates = cum_ret[mask_before_end]
        peak_indices = candidates[candidates == peak_value]

        if len(peak_indices) > 0:
            max_dd_start = peak_indices.index[-1]  # 最后一次达到peak的时间
        else:
            max_dd_start = ser.index[0]

        return [abs(max_dd), max_dd_start, max_dd_end]

    def generate_report(self, perf: pd.DataFrame, title: str = "Performance Report"):
        """
        生成格式化的控制台报告

        :param perf: 绩效统计 DataFrame
        :type perf: pd.DataFrame
        :param title: 报告标题, defaults to "Performance Report"
        :type title: str
        """
        # 整理显示用的 DataFrame
        report_df = pd.DataFrame(index=perf.index)
        report_df.index.name = "Period"

        # 1. 年化收益
        report_df["Ret(%)"] = perf["ret_mean"] * self.ann_factor * 100

        # 2. IC 相关
        if "ic_mean" in perf.columns:
            report_df["IC"] = perf["ic_mean"]
            report_df["IR"] = perf["ir"]

        # 3. Sharpe
        report_df["Sharpe"] = perf["sharpe"] * np.sqrt(self.ann_factor)

        # 4. MaxDD
        report_df["MaxDD(%)"] = perf["max_dd"] * 100

        # 5. Turnover
        if "turnover" in perf.columns:
            report_df["Tvr(%)"] = perf["turnover"] * 100

        # 6. Win Rate
        report_df["Win(%)"] = perf["win_rate"] * 100

        # 7. DD Range
        def _fmt_dd_range(row):
            s = row.get("max_dd_start")
            e = row.get("max_dd_end")
            if pd.notnull(s) and pd.notnull(e):
                return f"{s.strftime('%m%d')}-{e.strftime('%m%d')}"
            return "-"

        report_df["DD Range"] = perf.apply(_fmt_dd_range, axis=1)

        # 格式化浮点数 (虽然 pretty_table_print 也会处理，这里可以先做一些特定格式)
        # 比如百分比保留2位，IC保留3位
        # 但为了通用性，我们交给 pretty_table_print 处理或者在这里 round

        # Rounding for display
        report_df["Ret(%)"] = report_df["Ret(%)"].round(2)
        if "IC" in report_df.columns:
            report_df["IC"] = report_df["IC"].round(3)
            report_df["IR"] = report_df["IR"].round(3)
        report_df["Sharpe"] = report_df["Sharpe"].round(2)
        report_df["MaxDD(%)"] = report_df["MaxDD(%)"].round(2)
        if "Tvr(%)" in report_df.columns:
            report_df["Tvr(%)"] = report_df["Tvr(%)"].round(2)
        report_df["Win(%)"] = report_df["Win(%)"].round(1)
        return report_df


# ==========================================
# # 5. 因子分析器 (分析入口层 - 统一接层)
# ==========================================
class FactorAnalyzer:
    """
    因子分析大类 (Factor Analyzer)
    整合 FactorGrouper, VectorExecutor, PerformanceMetrics，提供一站式分析接口。

    :param n_groups: 分组数量, 默认为 5
    :param ascending: 分组排序方向, 默认为 False
    :param ignore_zeros: 分组时是否忽略 0 值, 默认为 False
    :param booksize: 目标市值（单边规模）, 默认为 1000000.0
    :param cost: 交易手续费率, 默认为 0.0
    :param filter_limit: 是否过滤涨跌停限制, 默认为 False
    :param filter_suspend: 是否过滤停牌限制, 默认为 False
    :param ic_method: IC 计算方法, 默认为 "spearman"
    :param delay: 因子信号生效延迟天数, 默认为 0
    :param ls_balance: 是否开启多空平衡模式 (LS Balance), 默认为 False
    :param periods_per_year: 年化周期对应的交易天数, 默认为 250
    :param factor: 因子值 DataFrame (可选，建议在初始化时传入), 默认为 None
    :param daily_ret: 日收益率数据 (可选, 若提供 settle_price 则可自动计算), 默认为 None
    :param trade_ret: 交易损耗/差异收益率 (可选, 若提供 trd_settle_price 则可自动计算), 默认为 None
    :param settle_price: 结算价格 (收盘价) 数据, 默认为 None
    :param trd_settle_price: 交易参考价格 (成交均价) 数据, 默认为 None
    :param adj_factor: 复权因子数据, 默认为 None
    :param limit_status: 涨跌停状态数据 (可选), 默认为 None
    :param suspend_status: 停牌状态数据 (可选), 默认为 None
    """

    def _ensure_tz_naive(self, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """确保 DataFrame 的 DatetimeIndex 是无时区的 (tz-naive)"""
        if (
            df is not None
            and isinstance(df.index, pd.DatetimeIndex)
            and df.index.tz is not None
        ):
            return df.copy().set_index(df.index.tz_localize(None))
        return df

    def __init__(
        self,
        # Grouper Args
        n_groups: int = 5,
        ascending: bool = False,
        ignore_zeros: bool = False,
        # Executor Args
        booksize: float = 1000000.0,
        cost: float = 0.0,
        filter_limit: bool = False,
        filter_suspend: bool = False,
        ic_method: str = "spearman",
        delay: int = 0,
        ls_balance: bool = False,
        # Metrics Args
        periods_per_year: int = 250,
        # Data Args
        factor: pd.DataFrame = None,
        daily_ret: pd.DataFrame = None,
        trade_ret: pd.DataFrame = None,
        settle_price: pd.DataFrame = None,
        trd_settle_price: pd.DataFrame = None,
        adj_factor: pd.DataFrame = None,
        limit_status: Optional[pd.DataFrame] = None,
        suspend_status: Optional[pd.DataFrame] = None,
    ):
        self.grouper = FactorGrouper(
            n_groups=n_groups, ascending=ascending, ignore_zeros=ignore_zeros
        )
        self.executor = VectorExecutor(
            booksize=booksize,
            cost=cost,
            filter_limit=filter_limit,
            filter_suspend=filter_suspend,
            ic_method=ic_method,
            delay=delay,
        )
        self.metrics = PerformanceMetrics(periods_per_year=periods_per_year)
        self.ls_balance = ls_balance

        # Data Sanitization: 确保所有输入数据的索引无时区，以保证对齐
        factor = self._ensure_tz_naive(factor)
        daily_ret = self._ensure_tz_naive(daily_ret)
        trade_ret = self._ensure_tz_naive(trade_ret)
        settle_price = self._ensure_tz_naive(settle_price)
        trd_settle_price = self._ensure_tz_naive(trd_settle_price)
        adj_factor = self._ensure_tz_naive(adj_factor)
        limit_status = self._ensure_tz_naive(limit_status)
        suspend_status = self._ensure_tz_naive(suspend_status)

        # 检查 factor 索引类型，防止传入了转置的数据 (Stock as Index)
        if factor is not None and not isinstance(factor.index, pd.DatetimeIndex):
            logger.warning(
                "Warning: Factor index is NOT DatetimeIndex. "
                "FactorAnalyzer Expects (Date x Stock) shape with DatetimeIndex. "
                "Please check if your factor DataFrame is transposed (e.g. use .unstack(level='instrument') if MultiIndex)."
            )

        self.factor = factor
        self.daily_ret = daily_ret
        self.trade_ret = trade_ret
        self.ic_ret = None  # 用于IC计算的收益率 (默认等于daily_ret，若有trd_price则使用trd_price的ret)
        self.limit_status = limit_status
        self.suspend_status = suspend_status

        # 自动计算收益率
        self._compute_returns_if_needed(settle_price, trd_settle_price, adj_factor)

    def _compute_returns_if_needed(self, settle_price, trd_settle_price, adj_factor):
        """
        如果提供了价格数据，自动计算收益率
        完全对齐原始 factor_test.py 的数据处理逻辑
        """
        if settle_price is not None and adj_factor is not None:
            # 1. 计算复权结算价（原有stat: adjsettle = settle_price * adjfactor）
            adj_settle = settle_price * adj_factor

            # 2. 计算日收益率 (Close-to-Close)（原有stat: daily_adjreturns = adjsettle.pct_change()）
            if self.daily_ret is None:
                self.daily_ret = adj_settle.pct_change()

            # 3. 处理交易价格
            if trd_settle_price is not None:
                adj_trd = trd_settle_price * adj_factor

                # 计算交易差异收益率 (Trade Price / Settle Price - 1)
                # 原有stat: trd_daily_adjreturns = trd_adjsettle.div(adjsettle) - 1
                if self.trade_ret is None:
                    self.trade_ret = (adj_trd / adj_settle) - 1

                # 计算用于IC的收益率 (Trade-to-Trade) - 这里保持与原始stat一致
                # 原有stat: trd_adjreturns = trd_adjsettle.pct_change()
                # IC计算使用这个收益率
                self.ic_ret = adj_trd.pct_change()
            else:
                # 没有交易价时，IC使用日收益率
                self.ic_ret = self.daily_ret

        # Fallback: 如果没有提供IC专用收益率，使用日收益率
        if self.ic_ret is None and self.daily_ret is not None:
            self.ic_ret = self.daily_ret

    def run_analysis(
        self,
        factor: Optional[pd.DataFrame] = None,
        daily_ret: Optional[pd.DataFrame] = None,
        trade_ret: Optional[pd.DataFrame] = None,
        limit_status: Optional[pd.DataFrame] = None,
        suspend_status: Optional[pd.DataFrame] = None,
        report_type: str = "yearly",
        ls_balance: bool = None,
        # Price Inputs for on-the-fly calc
        settle_price: pd.DataFrame = None,
        trd_settle_price: pd.DataFrame = None,
        adj_factor: pd.DataFrame = None,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
        """
        运行完整分析 (通常用于分析原始因子的整体表现)

        :param factor: 因子数据 (覆盖初始化参数), 默认为 None
        :param daily_ret: 日收益率数据 (覆盖初始化参数), 默认为 None
        :param trade_ret: 交易收益率数据 (覆盖初始化参数), 默认为 None
        :param limit_status: 涨跌停状态 (覆盖初始化参数), 默认为 None
        :param suspend_status: 停牌状态 (覆盖初始化参数), 默认为 None
        :param report_type: 报告聚合周期, 默认为 "yearly"
        :param ls_balance: 是否开启多空平衡 (覆盖初始化参数), 默认为 None
        :param settle_price: 结算价格 (覆盖初始化参数, 若计算 ret 则需要), 默认为 None
        :param trd_settle_price: 交易价格 (覆盖初始化参数, 若计算 ret 则需要), 默认为 None
        :param adj_factor: 复权因子 (覆盖初始化参数, 若计算 ret 则需要), 默认为 None
        :param verbose: 是否打印分析报告, 默认为 True
        :return: (PNL DataFrame, PNL Detail Dict, Performance DataFrame)
        :rtype: Tuple[pd.DataFrame, Dict, pd.DataFrame]
        """
        # Ensure Inputs are naive
        factor = self._ensure_tz_naive(factor)
        daily_ret = self._ensure_tz_naive(daily_ret)
        trade_ret = self._ensure_tz_naive(trade_ret)
        limit_status = self._ensure_tz_naive(limit_status)
        suspend_status = self._ensure_tz_naive(suspend_status)
        settle_price = self._ensure_tz_naive(settle_price)
        trd_settle_price = self._ensure_tz_naive(trd_settle_price)
        adj_factor = self._ensure_tz_naive(adj_factor)

        # Data resolution (args > self > error)
        factor = factor if factor is not None else self.factor
        ls_balance = ls_balance if ls_balance is not None else self.ls_balance

        # 处理收益率数据或从价格数据中瞬时生成
        # 如果提供了价格参数作为覆盖，则优先使用瞬时计算值
        local_daily_ret = daily_ret if daily_ret is not None else self.daily_ret
        local_trade_ret = trade_ret if trade_ret is not None else self.trade_ret
        local_ic_ret = self.ic_ret if self.ic_ret is not None else local_daily_ret

        # 若方法调用中显式传入了结算价格与复权因子，则重新计算核心收益序列
        if settle_price is not None and adj_factor is not None:
            adj_settle = settle_price * adj_factor
            local_daily_ret = adj_settle.pct_change()
            local_ic_ret = local_daily_ret

            if trd_settle_price is not None:
                adj_trd = trd_settle_price * adj_factor
                local_trade_ret = (adj_trd / adj_settle) - 1
                local_ic_ret = adj_trd.pct_change()

        # 数据完整性最终校验
        if factor is None or local_daily_ret is None:
            raise ValueError(
                "缺少必需的数据项 (factor, daily_ret)。请直接提供收益率数据或提供价格数据 (settle_price, adj_factor) 进行计算。"
            )

        # Optional: trade_ret defaulting
        if local_trade_ret is None:
            # logger.warning("未提供 trade_ret 参数，回测将假设交易价格为当日收盘价（0 滑点）。")
            pass

        # 0. 处理delay参数 - 按原始 factor_test.py 的逻辑
        # 原有stat在主流程中对position做了delay偏移：position = position.shift(delay)
        # 这里我们在VectorExecutor中已经处理了，但要确保与原始逻辑一致
        if self.executor.delay != 0:
            # delay已经在VectorExecutor.execute中处理，这里不需要额外处理
            pass

        # 1. 多空平衡 (LS Balance)
        if ls_balance:
            # 减去截面中位数，使因子以0为中心
            factor = factor.sub(factor.median(axis=1), axis=0)

        # 1. 计算IC (使用专门的 ic_ret)
        # 确保使用正确的收益率数据进行IC计算：
        # - 如果有交易价格数据，使用交易价收益率 (trd_adjreturns)
        # - 否则使用日收益率 (daily_adjreturns)
        ic_ret_for_calc = (
            local_ic_ret
            if hasattr(self, "ic_ret") and self.ic_ret is not None
            else local_daily_ret
        )
        ic = self.executor.calculate_information_coefficient(factor, ic_ret_for_calc)

        # 2. 回测 (将因子直接视为权重，内部会归一化)
        pnl_df, pnl_detail = self.executor.calculate_portfolio_returns(
            target_weights=factor,
            daily_ret=local_daily_ret,
            trade_ret=local_trade_ret,
            limit_status=limit_status,
            suspend_status=suspend_status,
            return_detail=True,
        )
        pnl_df["ic"] = ic

        # 3. 统计
        perf_df = self.metrics.calculate_by_period(pnl_df, ic, report_type=report_type)

        # 4. 报告
        if verbose:
            self.metrics.generate_report(perf_df, title="Factor Analysis Report")

        return pnl_df, pnl_detail, perf_df

    def calc_quantile(
        self,
        factor: Optional[pd.DataFrame] = None,
        daily_ret: Optional[pd.DataFrame] = None,
        trade_ret: Optional[pd.DataFrame] = None,
        limit_status: Optional[pd.DataFrame] = None,
        suspend_status: Optional[pd.DataFrame] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        计算分层回测 (Quantile Analysis)

        :param factor: 因子数据 (覆盖初始化参数), 默认为 None
        :param daily_ret: 日收益率数据 (覆盖初始化参数), 默认为 None
        :param trade_ret: 交易收益率数据 (覆盖初始化参数), 默认为 None
        :param limit_status: 涨跌停状态 (覆盖初始化参数), 默认为 None
        :param suspend_status: 停牌状态 (覆盖初始化参数), 默认为 None
        :return: 分层回测结果映射字典 {group_id: pnl_df}，包含多空对冲组合 "LS"
        :rtype: Dict[str, pd.DataFrame]
        """
        # Data resolution (args > self > error)
        factor = factor if factor is not None else self.factor
        daily_ret = daily_ret if daily_ret is not None else self.daily_ret
        trade_ret = trade_ret if trade_ret is not None else self.trade_ret
        limit_status = limit_status if limit_status is not None else self.limit_status
        suspend_status = (
            suspend_status if suspend_status is not None else self.suspend_status
        )

        if factor is None or daily_ret is None:
            raise ValueError(
                "缺少必需数据。请在初始化或方法调用时提供因子数据和收益数据。"
            )
        portfolios = self.grouper.get_quantile_groups(factor)
        results = {}

        logger.info(f"Starting Quantile Analysis (Groups: {self.grouper.n_groups})...")

        for gid, weights in portfolios.items():
            # 回测每一层
            pnl_df = self.executor.calculate_portfolio_returns(
                target_weights=weights,
                daily_ret=daily_ret,
                trade_ret=trade_ret,
                limit_status=limit_status,
                suspend_status=suspend_status,
                return_detail=False,
            )
            # 简单统计可以做，或者只返回 pnl_df
            results[gid] = pnl_df

        # 多空对冲 (Top - Bottom)
        # ascending=False (默认), Group 1=Max, Group N=Min
        # Long Group 1, Short Group N
        top_g = 1
        btm_g = self.grouper.n_groups

        ls_w = portfolios[top_g] - portfolios[btm_g]
        ls_pnl = self.executor.calculate_portfolio_returns(
            target_weights=ls_w,
            daily_ret=daily_ret,
            trade_ret=trade_ret,
            limit_status=limit_status,
            suspend_status=suspend_status,
        )
        results["LS"] = ls_pnl

        return results

    # ==========================================
    # 兼容性包装与便捷调用接口
    # ==========================================

    def performance_stat_by_period(
        self,
        result_df: pd.DataFrame,
        ic: Optional[pd.Series] = None,
        benchmark_ret: Optional[pd.Series] = None,
        report_type: str = "yearly",
    ) -> pd.DataFrame:
        """
        按周期聚合统计绩效 (兼容性包装层)

        :param result_df: 回测结果 DataFrame
        :type result_df: pd.DataFrame
        :param ic: IC 序列, 默认为 None
        :type ic: Optional[pd.Series]
        :param benchmark_ret: 基准收益率, 默认为 None
        :type benchmark_ret: Optional[pd.Series]
        :param report_type: 报告聚合类型, 默认为 "yearly"
        :type report_type: str
        :return: 绩效统计 DataFrame
        :rtype: pd.DataFrame
        """
        return self.metrics.calculate_by_period(
            result_df, ic, benchmark_ret, report_type
        )

    def generate_report(
        self, perf: pd.DataFrame, title: str = "Performance Report", use_df: bool = True
    ):
        """
        生成格式化的文本控制台报告 (兼容性包装层)

        :param perf: 绩效统计 DataFrame
        :type perf: pd.DataFrame
        :param title: 报告标题, defaults to "Performance Report"
        :type title: str
        """
        report = self.metrics.generate_report(perf=perf, title=title)
        # Call standalone pretty print function
        pretty_table_print(report, title=title)


# ==========================================
# 6. 主入口函数和数据接口适配层 (Data Interface Adapter Layer)
# ==========================================


def create_factor_analyzer_from_dataframes(
    factor_df: pd.DataFrame,
    settle_df: pd.DataFrame,
    adjfactor_df: pd.DataFrame,
    trd_settle_df: Optional[pd.DataFrame] = None,
    tradestatuscode_df: Optional[pd.DataFrame] = None,
    up_down_limit_status_df: Optional[pd.DataFrame] = None,
    # 分析参数
    n_groups: int = 5,
    ascending: bool = False,
    booksize: float = 1000000.0,
    cost: float = 0.0,
    filter_limit: bool = False,
    filter_suspend: bool = False,
    delay: int = 0,
    ls_balance: bool = False,
    periods_per_year: int = 250,
) -> FactorAnalyzer:
    """
    从标准DataFrame创建FactorAnalyzer实例
    完全解耦数据获取依赖，只接受标准格式的DataFrame输入

    :param factor_df: 因子值DataFrame (index=date, columns=stock_code)
    :type factor_df: pd.DataFrame
    :param settle_df: 结算价DataFrame (index=date, columns=stock_code)
    :type settle_df: pd.DataFrame
    :param adjfactor_df: 复权因子DataFrame (index=date, columns=stock_code)
    :type adjfactor_df: pd.DataFrame
    :param trd_settle_df: 交易参考价DataFrame (可选, index=date, columns=stock_code), 默认为 None
    :type trd_settle_df: Optional[pd.DataFrame]
    :param tradestatuscode_df: 交易状态DataFrame (可选, 1=停牌, 0=正常), 默认为 None
    :type tradestatuscode_df: Optional[pd.DataFrame]
    :param up_down_limit_status_df: 涨跌停状态DataFrame (可选, 1=涨停, -1=跌停, 0=正常), 默认为 None
    :type up_down_limit_status_df: Optional[pd.DataFrame]
    :param n_groups: 分层数量, 默认为 5
    :type n_groups: int
    :param ascending: 分层排序方向 (False: 因子大值为Group1), 默认为 False
    :type ascending: bool
    :param booksize: 目标市值, 默认为 1000000.0
    :type booksize: float
    :param cost: 交易成本率, 默认为 0.0
    :type cost: float
    :param filter_limit: 是否过滤涨跌停, 默认为 False
    :type filter_limit: bool
    :param filter_suspend: 是否过滤停牌, 默认为 False
    :type filter_suspend: bool
    :param delay: 因子生效延迟天数, 默认为 0
    :type delay: int
    :param ls_balance: 是否多空平衡, 默认为 False
    :type ls_balance: bool
    :param periods_per_year: 年化交易天数, 默认为 250
    :type periods_per_year: int
    :return: 配置好的FactorAnalyzer实例
    :rtype: FactorAnalyzer
    :raises TypeError: 当输入参数类型不正确时
    """

    # ========================================
    # 数据格式验证和预处理
    # ========================================

    # 1. 基本数据格式验证
    for name, df in [
        ("factor_df", factor_df),
        ("settle_df", settle_df),
        ("adjfactor_df", adjfactor_df),
    ]:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(
                f"{name} index must be DatetimeIndex. Please check if your input is transposed (Stocks as Index)."
            )

    # helper to ensure naive
    def ensure_naive(df):
        if (
            df is not None
            and isinstance(df.index, pd.DatetimeIndex)
            and df.index.tz is not None
        ):
            return df.copy().set_index(df.index.tz_localize(None))
        return df

    factor_df = ensure_naive(factor_df)
    settle_df = ensure_naive(settle_df)
    adjfactor_df = ensure_naive(adjfactor_df)
    trd_settle_df = ensure_naive(trd_settle_df)
    tradestatuscode_df = ensure_naive(tradestatuscode_df)
    up_down_limit_status_df = ensure_naive(up_down_limit_status_df)

    # 2. 计算复权收盘价和收益率（对应原有stat的adjsettle和daily_adjreturns）
    adj_settle = settle_df * adjfactor_df
    daily_ret = adj_settle.pct_change()

    # 3. 处理交易价格和交易收益差异（对应原有stat的trd_adjsettle和trd_daily_adjreturns）
    trade_ret = None
    ic_ret = daily_ret  # IC计算用的收益率
    if trd_settle_df is not None:
        adj_trd_settle = trd_settle_df * adjfactor_df
        # 交易差异收益率：原有stat逻辑 trd_daily_adjreturns = trd_adjsettle.div(adjsettle) - 1
        trade_ret = (adj_trd_settle / adj_settle) - 1
        # IC计算使用交易价收益率：原有stat逻辑 trd_adjreturns = trd_adjsettle.pct_change()
        ic_ret = adj_trd_settle.pct_change()

    # 4. 处理停牌状态数据（对应原有stat的suspension_status）
    suspend_status = None
    if filter_suspend and tradestatuscode_df is not None:
        # 原有stat逻辑：1表示停牌，0表示正常交易
        suspend_status = tradestatuscode_df

    # 5. 处理涨跌停状态数据（对应原有stat的up_down_limit_status）
    limit_status = None
    if filter_limit and up_down_limit_status_df is not None:
        # 原有stat逻辑：1=涨停, -1=跌停, 0=正常
        limit_status = up_down_limit_status_df

    # ========================================
    # 创建分析器实例
    # ========================================

    analyzer = FactorAnalyzer(
        n_groups=n_groups,
        ascending=ascending,
        booksize=booksize,
        cost=cost,
        filter_limit=filter_limit,
        filter_suspend=filter_suspend,
        delay=delay,
        ls_balance=ls_balance,
        periods_per_year=periods_per_year,
        # 数据参数
        factor=factor_df,
        daily_ret=daily_ret,
        trade_ret=trade_ret,
        limit_status=limit_status,
        suspend_status=suspend_status,
    )

    # 设置IC计算专用收益率
    analyzer.ic_ret = ic_ret

    return analyzer


def run_factor_analysis(
    factor_df: pd.DataFrame,
    settle_df: pd.DataFrame,
    adjfactor_df: pd.DataFrame,
    trd_settle_df: Optional[pd.DataFrame] = None,
    tradestatuscode_df: Optional[pd.DataFrame] = None,
    up_down_limit_status_df: Optional[pd.DataFrame] = None,
    # 分析参数
    analysis_type: str = "full",  # "full" 或 "quantile"
    n_groups: int = 5,
    ascending: bool = False,
    booksize: float = 1000000.0,
    cost: float = 0.0,
    filter_limit: bool = False,
    filter_suspend: bool = False,
    delay: int = 0,
    ls_balance: bool = False,
    report_type: str = "yearly",
    verbose: bool = True,
    periods_per_year: int = 250,
) -> Union[Tuple[pd.DataFrame, Dict, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    一键运行因子分析
    完全替代原有Stats类，提供相同的分析能力但使用DataFrame输入

    :param factor_df: 因子值DataFrame
    :type factor_df: pd.DataFrame
    :param settle_df: 结算价DataFrame
    :type settle_df: pd.DataFrame
    :param adjfactor_df: 复权因子DataFrame
    :type adjfactor_df: pd.DataFrame
    :param trd_settle_df: 交易参考价DataFrame (可选), 默认为 None
    :type trd_settle_df: Optional[pd.DataFrame]
    :param tradestatuscode_df: 交易状态DataFrame (可选), 默认为 None
    :type tradestatuscode_df: Optional[pd.DataFrame]
    :param up_down_limit_status_df: 涨跌停状态DataFrame (可选), 默认为 None
    :type up_down_limit_status_df: Optional[pd.DataFrame]
    :param analysis_type: 分析类型 ("full"=整体分析, "quantile"=分层分析), 默认为 "full"
    :type analysis_type: str
    :param n_groups: 分层数量, 默认为 5
    :type n_groups: int
    :param ascending: 分层排序方向, 默认为 False
    :type ascending: bool
    :param booksize: 目标市值, 默认为 1000000.0
    :type booksize: float
    :param cost: 交易成本率, 默认为 0.0
    :type cost: float
    :param filter_limit: 是否过滤涨跌停, 默认为 False
    :type filter_limit: bool
    :param filter_suspend: 是否过滤停牌, 默认为 False
    :type filter_suspend: bool
    :param delay: 因子生效延迟天数, 默认为 0
    :type delay: int
    :param ls_balance: 是否多空平衡, 默认为 False
    :type ls_balance: bool
    :param report_type: 报告聚合周期, 默认为 "yearly"
    :type report_type: str
    :param verbose: 是否打印报告, 默认为 True
    :type verbose: bool
    :param periods_per_year: 年化交易天数, 默认为 250
    :type periods_per_year: int
    :return: 分析结果 (full: (pnl_df, pnl_detail, perf_df), quantile: {group_id: pnl_df})
    :rtype: Union[Tuple[pd.DataFrame, Dict, pd.DataFrame], Dict[str, pd.DataFrame]]
    :raises ValueError: 当analysis_type不支持时
    """

    # 创建分析器
    analyzer = create_factor_analyzer_from_dataframes(
        factor_df=factor_df,
        settle_df=settle_df,
        adjfactor_df=adjfactor_df,
        trd_settle_df=trd_settle_df,
        tradestatuscode_df=tradestatuscode_df,
        up_down_limit_status_df=up_down_limit_status_df,
        n_groups=n_groups,
        ascending=ascending,
        booksize=booksize,
        cost=cost,
        filter_limit=filter_limit,
        filter_suspend=filter_suspend,
        delay=delay,
        ls_balance=ls_balance,
        periods_per_year=periods_per_year,
    )

    if analysis_type == "full":
        # 整体分析 - 对应原有 Stats.calc()
        return analyzer.run_analysis(report_type=report_type, verbose=verbose)
    elif analysis_type == "quantile":
        # 分层分析 - 对应原有 Stats.calc_quantile()
        return analyzer.calc_quantile()
    else:
        raise ValueError(
            f"Unknown analysis_type: {analysis_type}. Must be 'full' or 'quantile'"
        )
