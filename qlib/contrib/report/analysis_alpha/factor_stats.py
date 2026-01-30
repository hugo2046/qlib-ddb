import pandas as pd
import numpy as np
from loguru import logger
from typing import Optional, Dict, List, Union, Tuple
from datetime import datetime

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
        ascending: bool = False,  # 默认降序，Group 1为最大值
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

        # 质量检查
        self._check_data_quality(work_factor)

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
            weights = mask.astype(float).div(mask.sum(axis=1), axis=0).fillna(0)

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

        return portfolios

    def _check_data_quality(self, df: pd.DataFrame):
        valid_count = df.count(axis=1)
        # 硬性检查
        if (valid_count < self.n_groups).any():
            logger.warning(
                f"Warning: Some dates have insufficient valid data to form {self.n_groups} groups."
            )
        # 软性检查
        nan_rates = 1.0 - (valid_count / df.shape[1])
        if (nan_rates > self.max_nan_rate).any():
            msg = f"High NaN Rate Warning: Max NaN rate is {nan_rates.max():.1%}."
            if self.strict_mode:
                raise DataQualityError(msg)
            else:
                logger.warning(msg)


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

    def calc_ic(
        self, factor: pd.DataFrame, returns: pd.DataFrame, method: str = None
    ) -> pd.Series:
        """
        计算 IC (信息系数)

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

        # 1. 对齐收益率 (T+1 收益，并充分考虑延迟 delay 参数)
        # 假设 factor 在 T 日，预测 T+delay 日之后的收益
        # 通常 IC 是 T日因子 vs T+1日收益
        # 如果 delay=0，则使用 factor[t] 对应 returns[t] (由于行情数据多指 T 收盘，通常 T+1 更有预测意义)
        # 如果 delay=1 (默认 T+1)，则不进行 shift (0) 相关性计算
        # 原有stat逻辑: stock_return = trd_adjreturns.shift(delay - 1)
        # 如果 delay=1 (默认T+1)，shift(0)，即使用当天的 return 序列（注意：return的timestamp定义很重要）
        # 假设 returns 的 index t 代表 t日的收益（(P_t - P_{t-1})/P_{t-1}）
        # 则 T日的factor 应该和 T+1日的 returns 进行相关性分析
        # returns.shift(-1) 将 T+1 的收益移到 T

        # 兼容性处理：若外部直接传入了对其好的 returns，则不需要 shift
        # 这里遵循标准逻辑：factor[t] corr returns[t+delay]
        # 所以我们需要把 returns 向以前 shift delay 天
        # ret_aligned = returns.shift(-self.delay) # 暂时简化处理，假设传入的returns已经是对齐好的或者不需要内部shift复杂逻辑
        # 原有stat原逻辑是：trd_adjreturns.shift(self.delay - 1)
        # 如果 delay=1, shift(0). 也就是 factor[t] vs returns[t]
        # 这意味着 原有stat 中 returns[t] 存储的是 T+1 的收益？或者 factor[t] 是 T+1 的因子？
        # 通常回测框架：returns[t]是 T-1到T的收益。
        # 因子分析：Factor[t] (闭市后) vs Returns[t+1] (T到T+1)
        # 所以 Returns[t+1] 应该对齐到 Index t

        # 为避免歧义，这里采用最直接的方式：
        # 我们假设传入的 daily_ret 索引 t 代表 t日的收益
        # 我们需要计算 corr(Factor[t], Return[t+1])
        # 所以将 Return 向上移动 1位 (shift(-1))

        ret_aligned = returns.shift(-1)  # 默认 T+1 IC

        # 仅保留因子非空的位置
        valid_mask = factor.notna() & ret_aligned.notna()

        # 计算IC
        ic = factor[valid_mask].corrwith(ret_aligned[valid_mask], axis=1, method=method)
        return ic

    def execute(
        self,
        target_weights: pd.DataFrame,
        daily_ret: pd.DataFrame,
        trade_ret: pd.DataFrame,
        limit_status: Optional[pd.DataFrame] = None,
        suspend_status: Optional[pd.DataFrame] = None,
        return_detail: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
        """
        执行向量化回测

        :param target_weights: 目标持仓权重
        :type target_weights: pd.DataFrame
        :param daily_ret: 日收益率
        :type daily_ret: pd.DataFrame
        :param trade_ret: 交易收益率 (用于计算交易期间的差异或损耗)
        :type trade_ret: pd.DataFrame
        :param limit_status: 涨跌停状态 (1: 涨停, -1: 跌停, 0: 正常), 默认为 None
        :type limit_status: Optional[pd.DataFrame]
        :param suspend_status: 停牌状态 (1: 停牌, 0: 正常), 默认为 None
        :type suspend_status: Optional[pd.DataFrame]
        :param return_detail: 是否返回详细收益 (PNL) 信息序列, 默认为 False
        :type return_detail: bool
        :return: 回测结果 DataFrame, 或 (pnl_df, pnl_detail) 元组
        :rtype: Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]
        """
        # 复刻 原有stat pnl_calc 的核心逻辑

        # 1. 基础处理
        position = target_weights.copy()

        # 处理 Delay (Factor[t] -> Position[t+delay])
        if self.delay != 0:
            position = position.shift(self.delay)

        position_adj_num = 0.000001

        # 空值填充
        position.fillna(0, inplace=True)

        # 2. 市值归一化处理 (资金分配规模)
        position_sum = position.abs().sum(axis=1) + position_adj_num
        daily_value = position_sum

        if self.booksize > 0:
            # 归一化到目标市值
            book_unit = position_sum.rtruediv(self.booksize)
            position = position.multiply(book_unit, axis="index")
            daily_value = pd.Series(self.booksize, index=position.index)

        position_sum_before_filter = position.abs().sum(axis=1)

        # 3. 交易限制过滤逻辑 (涨跌停/停牌)
        # 过滤涨跌停
        if self.filter_limit and limit_status is not None:
            # limit_status: 1=涨停, -1=跌停
            # position_change > 0 (加仓) & limit=1 (涨停) -> 禁止买入涨停
            # position_change < 0 (减仓) & limit=-1 (跌停) -> 禁止卖出跌停

            # 为了计算 change，我们需要 pre_position
            # 但这里是一次性计算，position 是目标仓位
            # 原有stat 逻辑：position_change = position.diff()
            # 这是一个近似，因为实际仓位可能受以前的过滤影响
            # 在向量化中，很难做到完全的路径依赖过滤，除非循环
            # 原有stat 的做法是先 diff，然后 mask，然后 scale

            curr_pos_diff = position.diff().fillna(0)

            # 若 limit_status 对齐
            status = limit_status.reindex(
                index=position.index, columns=position.columns
            ).fillna(0)

            # 假设 position 正值代表多头
            # 如果做多(Pos>0):
            #   加仓(Diff>0) 且 涨停(Status=1) -> 禁止
            #   减仓(Diff<0) 且 跌停(Status=-1) -> 禁止

            # 多头视角
            cant_buy = (curr_pos_diff > 0) & (status == 1)
            cant_sell = (curr_pos_diff < 0) & (status == -1)

            # 空头视角 (Pos<0)
            #   加仓(变更多空/Diff<0) 且 ... 逻辑比较复杂，原有stat主要处理多头或简单逻辑
            #   原有stat: up_limit = status == 1 * sign; down_limit = status == -1 * sign
            #   这里简化处理，假设 status 1 表示 价格涨停，-1 表示 价格跌停
            #   无论多空：
            #     不能以涨停价买入 (Buy: Pos增加) -> Diff > 0 & Status == 1
            #     不能以跌停价卖出 (Sell: Pos减少) -> Diff < 0 & Status == -1

            mask = cant_buy | cant_sell
            position[mask] = np.nan  # 标记为无效，稍后处理

        # 过滤停牌
        if self.filter_suspend and suspend_status is not None:
            status = suspend_status.reindex(
                index=position.index, columns=position.columns
            ).fillna(0)
            # 1 表示停牌
            position[status == 1] = np.nan

        # 4. 仓位无效值回填与资本杠杆缩放 (Scaling)
        # 原有stat: ffill nan, then fillna(0)
        position.ffill(inplace=True)
        position.fillna(0, inplace=True)

        position_sum_after_filter = position.abs().sum(axis=1) + position_adj_num
        scale_final = position_sum_before_filter.div(position_sum_after_filter)
        position = position.mul(scale_final, axis=0)

        # 5. 回测收益指标计算 (PnL 计算)
        daily_count = (position != 0).sum(axis=1)

        # 持仓收益 (Ins Pnl): T-1 持仓 * T日收益
        last_position = position.shift(1).fillna(0)
        daily_ins_pnl = last_position.multiply(daily_ret)

        # 交易执行产生的收益计算 (Trading)
        # 计算需要交易的金额
        if self.booksize > 0:
            daily_ins_trd = position.diff()
        else:
            # 考虑自然增长后的漂移
            # T-1市值 * (1+ret) = 交易前市值
            # 目标市值 - 交易前市值 = 需交易金额
            daily_ins_trd = position.sub(
                last_position.multiply(1 + daily_ret.shift(1).fillna(0))
            )

        daily_ins_trd.iloc[0] = position.iloc[0]  # 首日全额买入

        # 交易收益损耗 (Trd Pnl): -1 * 交易执行额 * 交易期间滑点差异(trade_ret)
        # trade_ret 通常定义为 (成交均价 - 收盘价) / 收盘价
        # 原有stat: daily_trd_pnl = -daily_ins_trd * trd_daily_adjreturns
        daily_trd_pnl = -daily_ins_trd.multiply(trade_ret)

        # 总收益
        daily_total_pnl = daily_ins_pnl.fillna(0) + daily_trd_pnl.fillna(0)

        # 聚合
        s_pnl = daily_total_pnl.sum(axis=1)
        s_trd = daily_ins_trd.abs().sum(axis=1)
        s_val = daily_value

        # 换手率
        # last_daily_value
        last_val = s_val.shift(1).replace(0, np.nan).bfill()
        s_tvr = s_trd.div(last_val)

        # 收益率 (扣费前)
        s_rawret = s_pnl.div(last_val)

        # 扣费
        if self.cost > 0:
            # 费用 = 交易额 * rate
            cost_val = s_trd * self.cost
            s_ret = (s_pnl - cost_val).div(last_val)
        else:
            s_ret = s_rawret

        # 结果汇总
        pnl_df = pd.DataFrame(
            {
                "ret": s_ret.fillna(0),
                "rawret": s_rawret.fillna(0),
                "trd": s_trd.fillna(0),
                "tvr": s_tvr.fillna(0),
                "value": s_val.fillna(0),
                "count": daily_count.fillna(0),
                "pnl": s_pnl.fillna(0),
                "ins_pnl": daily_ins_pnl.sum(axis=1).fillna(0),
                "trd_pnl": daily_trd_pnl.sum(axis=1).fillna(0),
            }
        )

        pnl_detail = {
            "daily_ins_pnl": daily_ins_pnl,
            "daily_trd_pnl": daily_trd_pnl,
            "daily_total_pnl": daily_total_pnl,
            "daily_ins_trd": daily_ins_trd,
        }

        if return_detail:
            return pnl_df, pnl_detail
        return pnl_df


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
        """计算最大回撤及起止时间"""
        if len(ser) == 0:
            return 0.0, None, None

        cum = (1 + ser).cumprod()
        max_cum = cum.expanding().max()
        dd = (cum / max_cum) - 1

        max_dd = dd.min()
        end_idx = dd.idxmin()

        # 寻找 start_idx: end_idx 之前的最高点
        # 截取到 end_idx
        sub_max_cum = max_cum[:end_idx]
        if len(sub_max_cum) == 0:
            start_idx = end_idx
        else:
            # 最高点的值
            peak_val = max_cum[end_idx]
            # 找到第一个达到这个peak的日期
            start_idx = sub_max_cum[sub_max_cum == peak_val].index[0]

        return abs(max_dd), start_idx, end_idx

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

        # Call standalone pretty print function
        pretty_table_print(report_df, title=title)


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

        # Remove timezone info if present to avoid warnings
        if (
            factor is not None
            and isinstance(factor.index, pd.DatetimeIndex)
            and factor.index.tz is not None
        ):
            factor.index = factor.index.tz_localize(None)

        self.factor = factor
        self.daily_ret = daily_ret
        self.trade_ret = trade_ret
        self.ic_ret = None  # 用于IC计算的收益率 (默认等于daily_ret，若有trd_price则使用trd_price的ret)
        self.limit_status = limit_status
        self.suspend_status = suspend_status

        # 自动计算收益率
        self._compute_returns_if_needed(settle_price, trd_settle_price, adj_factor)

    def _compute_returns_if_needed(self, settle_price, trd_settle_price, adj_factor):
        """如果提供了价格数据，自动计算收益率"""
        if settle_price is not None and adj_factor is not None:
            # 1. 计算复权结算价
            adj_settle = settle_price * adj_factor

            # 2. 计算日收益率 (Close-to-Close)
            if self.daily_ret is None:
                self.daily_ret = adj_settle.pct_change()

            # 3. 处理交易价格
            if trd_settle_price is not None:
                adj_trd = trd_settle_price * adj_factor

                # 计算交易差异收益率 (Trade Price / Settle Price - 1)
                # 原有stat: trd_daily_adjreturns = trd_adjsettle.div(adjsettle) - 1
                if self.trade_ret is None:
                    self.trade_ret = (adj_trd / adj_settle) - 1

                # 计算用于IC的收益率 (Trade-to-Trade)
                # 原有stat: trd_adjreturns = trd_adjsettle.pct_change()
                if self.ic_ret is None:
                    self.ic_ret = adj_trd.pct_change()

        # Fallback
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

        # 0. 多空平衡 (LS Balance)
        if ls_balance:
            # 减去截面中位数，使因子以0为中心
            factor = factor.sub(factor.median(axis=1), axis=0)

        # 1. 计算IC (使用 ic_ret)
        ic = self.executor.calc_ic(factor, local_ic_ret)

        # 2. 回测 (将因子直接视为权重，内部会归一化)
        pnl_df, pnl_detail = self.executor.execute(
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
        report_type: str = "yearly",
    ) -> Dict[str, pd.DataFrame]:
        """
        计算分层回测 (Quantile Analysis)

        :param factor: 因子数据 (覆盖初始化参数), 默认为 None
        :param daily_ret: 日收益率数据 (覆盖初始化参数), 默认为 None
        :param trade_ret: 交易收益率数据 (覆盖初始化参数), 默认为 None
        :param limit_status: 涨跌停状态 (覆盖初始化参数), 默认为 None
        :param suspend_status: 停牌状态 (覆盖初始化参数), 默认为 None
        :param report_type: (当前保留参数) 保留接口兼容性, 默认为 "yearly"
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

        if factor is None or daily_ret is None or trade_ret is None:
            raise ValueError(
                "缺少必需数据。请在初始化或方法调用时提供因数据和收益数据。"
            )
        portfolios = self.grouper.get_quantile_groups(factor)
        results = {}

        logger.info(f"Starting Quantile Analysis (Groups: {self.grouper.n_groups})...")

        for gid, weights in portfolios.items():
            # 回测每一层
            pnl_df = self.executor.execute(
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
        ls_pnl = self.executor.execute(
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

    def generate_report(self, perf: pd.DataFrame, title: str = "Performance Report"):
        """
        生成格式化的文本控制台报告 (兼容性包装层)

        :param perf: 绩效统计 DataFrame
        :type perf: pd.DataFrame
        :param title: 报告标题, defaults to "Performance Report"
        :type title: str
        """
        self.metrics.generate_report(perf, title)
