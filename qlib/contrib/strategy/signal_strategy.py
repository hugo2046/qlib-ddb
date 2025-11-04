# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import copy
import warnings
import numpy as np
import pandas as pd

from typing import Dict, List, Text, Tuple, Union
from abc import ABC

from qlib.data import D
from qlib.data.dataset import Dataset
from qlib.model.base import BaseModel
from qlib.strategy.base import BaseStrategy
from qlib.backtest.position import Position
from qlib.backtest.signal import Signal, create_signal_from
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.log import get_module_logger
from qlib.utils import get_pre_trading_date, load_dataset
from qlib.contrib.strategy.order_generator import OrderGenerator, OrderGenWOInteract
from qlib.contrib.strategy.optimizer import EnhancedIndexingOptimizer


# Constants for TopkRebalanceStrategy
MIN_TRADE_AMOUNT_THRESHOLD = 1e-6  # Minimum threshold for trade amount to avoid dust trades
DEFAULT_TRADE_UNIT_FACTOR = 1  # Default trade unit factor for round_amount_by_trade_unit


class BaseSignalStrategy(BaseStrategy, ABC):
    def __init__(
        self,
        *,
        signal: Union[Signal, Tuple[BaseModel, Dataset], List, Dict, Text, pd.Series, pd.DataFrame] = None,
        model=None,
        dataset=None,
        risk_degree: float = 0.95,
        trade_exchange=None,
        level_infra=None,
        common_infra=None,
        **kwargs,
    ):
        """
        Parameters
        -----------
        signal :
            the information to describe a signal. Please refer to the docs of `qlib.backtest.signal.create_signal_from`
            the decision of the strategy will base on the given signal
        risk_degree : float
            position percentage of total value.
        trade_exchange : Exchange
            exchange that provides market info, used to deal order and generate report
            - If `trade_exchange` is None, self.trade_exchange will be set with common_infra
            - It allowes different trade_exchanges is used in different executions.
            - For example:
                - In daily execution, both daily exchange and minutely are usable, but the daily exchange is recommended because it runs faster.
                - In minutely execution, the daily exchange is not usable, only the minutely exchange is recommended.

        """
        super().__init__(level_infra=level_infra, common_infra=common_infra, trade_exchange=trade_exchange, **kwargs)

        self.risk_degree = risk_degree

        # This is trying to be compatible with previous version of qlib task config
        if model is not None and dataset is not None:
            warnings.warn("`model` `dataset` is deprecated; use `signal`.", DeprecationWarning)
            signal = model, dataset

        self.signal: Signal = create_signal_from(signal)

    def get_risk_degree(self, trade_step=None):
        """get_risk_degree
        Return the proportion of your total value you will use in investment.
        Dynamically risk_degree will result in Market timing.
        """
        # It will use 95% amount of your total value by default
        return self.risk_degree


class TopkDropoutStrategy(BaseSignalStrategy):
    # TODO:
    # 1. Supporting leverage the get_range_limit result from the decision
    # 2. Supporting alter_outer_trade_decision
    # 3. Supporting checking the availability of trade decision
    # 4. Regenerate results with forbid_all_trade_at_limit set to false and flip the default to false, as it is consistent with reality.
    def __init__(
        self,
        *,
        topk,
        n_drop,
        method_sell="bottom",
        method_buy="top",
        hold_thresh=1,
        only_tradable=False,
        forbid_all_trade_at_limit=True,
        **kwargs,
    ):
        """
        Parameters
        -----------
        topk : int
            the number of stocks in the portfolio.
        n_drop : int
            number of stocks to be replaced in each trading date.
        method_sell : str
            dropout method_sell, random/bottom.
        method_buy : str
            dropout method_buy, random/top.
        hold_thresh : int
            minimum holding days
            before sell stock , will check current.get_stock_count(order.stock_id) >= self.hold_thresh.
        only_tradable : bool
            will the strategy only consider the tradable stock when buying and selling.

            if only_tradable:

                strategy will make decision with the tradable state of the stock info and avoid buy and sell them.

            else:

                strategy will make buy sell decision without checking the tradable state of the stock.
        forbid_all_trade_at_limit : bool
            if forbid all trades when limit_up or limit_down reached.

            if forbid_all_trade_at_limit:

                strategy will not do any trade when price reaches limit up/down, even not sell at limit up nor buy at
                limit down, though allowed in reality.

            else:

                strategy will sell at limit up and buy ad limit down.
        """
        super().__init__(**kwargs)
        self.topk = topk
        self.n_drop = n_drop
        self.method_sell = method_sell
        self.method_buy = method_buy
        self.hold_thresh = hold_thresh
        self.only_tradable = only_tradable
        self.forbid_all_trade_at_limit = forbid_all_trade_at_limit

    def generate_trade_decision(self, execute_result=None):
        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        # NOTE: the current version of topk dropout strategy can't handle pd.DataFrame(multiple signal)
        # So it only leverage the first col of signal
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None:
            return TradeDecisionWO([], self)
        if self.only_tradable:
            # If The strategy only consider tradable stock when make decision
            # It needs following actions to filter stocks
            def get_first_n(li, n, reverse=False):
                cur_n = 0
                res = []
                for si in reversed(li) if reverse else li:
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    ):
                        res.append(si)
                        cur_n += 1
                        if cur_n >= n:
                            break
                return res[::-1] if reverse else res

            def get_last_n(li, n):
                return get_first_n(li, n, reverse=True)

            def filter_stock(li):
                return [
                    si
                    for si in li
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    )
                ]

        else:
            # Otherwise, the stock will make decision without the stock tradable info
            def get_first_n(li, n):
                return list(li)[:n]

            def get_last_n(li, n):
                return list(li)[-n:]

            def filter_stock(li):
                return li

        current_temp: Position = copy.deepcopy(self.trade_position)
        # generate order list for this adjust date
        sell_order_list = []
        buy_order_list = []
        # load score
        cash = current_temp.get_cash()
        current_stock_list = current_temp.get_stock_list()
        # last position (sorted by score)
        last = pred_score.reindex(current_stock_list).sort_values(ascending=False).index
        # The new stocks today want to buy **at most**
        if self.method_buy == "top":
            # NOTE:如果score部分为[1,2,3,np.nan,np.nan],topk=5,n_drop=0时。则会把np.nan部分也纳入
            today = get_first_n(
                pred_score[~pred_score.index.isin(last)].sort_values(ascending=False).index,
                self.n_drop + self.topk - len(last),
            )
        elif self.method_buy == "random":
            topk_candi = get_first_n(pred_score.sort_values(ascending=False).index, self.topk)
            candi = list(filter(lambda x: x not in last, topk_candi))
            n = self.n_drop + self.topk - len(last)
            try:
                today = np.random.choice(candi, n, replace=False)
            except ValueError:
                today = candi
        else:
            raise NotImplementedError(f"This type of input is not supported")
        # combine(new stocks + last stocks),  we will drop stocks from this list
        # In case of dropping higher score stock and buying lower score stock.
        comb = pred_score.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index

        # Get the stock list we really want to sell (After filtering the case that we sell high and buy low)
        if self.method_sell == "bottom":
            sell = last[last.isin(get_last_n(comb, self.n_drop))]
        elif self.method_sell == "random":
            candi = filter_stock(last)
            try:
                sell = pd.Index(np.random.choice(candi, self.n_drop, replace=False) if len(last) else [])
            except ValueError:  # No enough candidates
                sell = candi
        else:
            raise NotImplementedError(f"This type of input is not supported")

        # Get the stock list we really want to buy
        buy = today[: len(sell) + self.topk - len(last)]
        # FIXME:如果score部分为[1,2,3,np.nan,np.nan],topk=5,n_drop=0时。则会把np.nan部分也纳入
        # 所以在这里建立验证过滤np.nan
        # valid_buy_scores = pred_score.reindex(buy).dropna()
        # buy = valid_buy_scores.index.tolist()
    
        for code in current_stock_list:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
            ):
                continue
            if code in sell:
                # check hold limit
                time_per_step = self.trade_calendar.get_freq()
                if current_temp.get_stock_count(code, bar=time_per_step) < self.hold_thresh:
                    continue
                # sell order
                sell_amount = current_temp.get_stock_amount(code=code)
                # sell_amount = self.trade_exchange.round_amount_by_trade_unit(sell_amount, factor)
                sell_order = Order(
                    stock_id=code,
                    amount=sell_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=Order.SELL,  # 0 for sell, 1 for buy
                )
                # is order executable
                if self.trade_exchange.check_order(sell_order):
                    sell_order_list.append(sell_order)
                    trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                        sell_order, position=current_temp
                    )
                    # update cash
                    cash += trade_val - trade_cost
        # buy new stock
        # note the current has been changed
        # current_stock_list = current_temp.get_stock_list()
        value = cash * self.risk_degree / len(buy) if len(buy) > 0 else 0

        # open_cost should be considered in the real trading environment, while the backtest in evaluate.py does not
        # consider it as the aim of demo is to accomplish same strategy as evaluate.py, so comment out this line
        # value = value / (1+self.trade_exchange.open_cost) # set open_cost limit
        for code in buy:
            # check is stock suspended
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.BUY,
            ):
                continue
            # buy order
            buy_price = self.trade_exchange.get_deal_price(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY
            )
            buy_amount = value / buy_price
            factor = self.trade_exchange.get_factor(stock_id=code, start_time=trade_start_time, end_time=trade_end_time)
            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
            buy_order = Order(
                stock_id=code,
                amount=buy_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.BUY,  # 1 for buy
            )
            buy_order_list.append(buy_order)
        return TradeDecisionWO(sell_order_list + buy_order_list, self)
    
class TopkRebalanceStrategy(BaseSignalStrategy):
    """Top-K Rebalance Strategy

    This strategy performs the following operations on each rebalancing date:
    1. Sort all stocks based on prediction signals.
    2. Select the topk or bottomk stocks with highest or lowest scores as target portfolio.
    3. Sell all stocks in current position that are not in the target portfolio.
    4. Adjust holdings to achieve equal weight for each stock in the target portfolio.

    Note: This strategy provides a cleaner approach compared to TopkDropoutStrategy
    by directly selecting topk stocks for equal weight configuration.
    """
    def __init__(self,
        *,
        topk,
        method_buy="top",
        hold_thresh=1,
        only_tradable=False,
        forbid_all_trade_at_limit=True,
        n_drop=None, # for backward compatibility
        **kwargs,)->None:
        """
        :param topk: int
            投资组合中的目标股票数量。
        :param method_buy: str
            买入方法，可以是 'top'（买入得分最高的）或 'bottom'（买入得分最低的）。
        :param hold_thresh: int
            最短持有天数。在卖出股票前，会检查 `current.get_stock_count(order.stock_id) >= self.hold_thresh`。
        :param only_tradable: bool
            买卖时是否只考虑可交易的股票。
            如果为 True，策略将根据股票的交易状态进行决策，避免买卖不可交易的股票。
            如果为 False，策略在决策时不会检查股票的交易状态。
        :param forbid_all_trade_at_limit: bool
            当价格达到涨跌停限制时，是否禁止所有交易。
            如果为 True，当价格达到涨跌停时，策略不会进行任何交易。
            如果为 False，策略将在涨停时卖出，在跌停时买入。
        :param n_drop: int, optional
            为保持向后兼容性而保留的参数，当前策略中未使用。
        """
        super().__init__(**kwargs)
        self.topk = topk
        self.method_buy = method_buy
        self.hold_thresh = hold_thresh
        self.only_tradable = only_tradable
        self.forbid_all_trade_at_limit = forbid_all_trade_at_limit

    def _get_tradable_stocks(self, stock_candidates, trade_start_time, trade_end_time):
        """
        Get tradable stocks from candidates based on tradability check.

        Parameters
        ----------
        stock_candidates : pd.Index
            Candidate stock IDs to check for tradability
        trade_start_time : pd.Timestamp
            Start time for tradability check
        trade_end_time : pd.Timestamp
            End time for tradability check

        Returns
        -------
        pd.Index
            Tradable stocks (subset of input candidates)
        """
        if not self.only_tradable:
            return stock_candidates[:self.topk]

        tradable_stocks = []
        for stock_id in stock_candidates:
            if self.trade_exchange.is_stock_tradable(
                stock_id=stock_id, start_time=trade_start_time, end_time=trade_end_time
            ):
                tradable_stocks.append(stock_id)
                if len(tradable_stocks) == self.topk:
                    break
        return pd.Index(tradable_stocks)

    def _is_stock_tradable_with_direction(self, stock_id, direction, trade_start_time, trade_end_time):
        """
        Check if a stock is tradable considering direction and limit restrictions.

        Parameters
        ----------
        stock_id : str
            Stock ID to check
        direction : OrderDir
            Trading direction (BUY/SELL)
        trade_start_time : pd.Timestamp
            Start time for tradability check
        trade_end_time : pd.Timestamp
            End time for tradability check

        Returns
        -------
        bool
            True if stock is tradable under given conditions
        """
        direction_param = None if self.forbid_all_trade_at_limit else direction
        return self.trade_exchange.is_stock_tradable(
            stock_id=stock_id,
            start_time=trade_start_time,
            end_time=trade_end_time,
            direction=direction_param,
        )

    def _create_order(self, stock_id, amount, direction, trade_start_time, trade_end_time):
        """
        Create and validate a trading order.

        Parameters
        ----------
        stock_id : str
            Stock ID to trade
        amount : float
            Trade amount
        direction : OrderDir
            Trading direction
        trade_start_time : pd.Timestamp
            Order start time
        trade_end_time : pd.Timestamp
            Order end time

        Returns
        -------
        Order or None
            Validated order or None if invalid
        """
        order = Order(
            stock_id=stock_id,
            amount=amount,
            start_time=trade_start_time,
            end_time=trade_end_time,
            direction=direction,
        )
        return order if self.trade_exchange.check_order(order) else None

    def _generate_sell_orders(self, sell_stocks, current_temp, trade_start_time, trade_end_time):
        """
        Generate sell orders for stocks that need to be sold.

        Parameters
        ----------
        sell_stocks : pd.Index
            Stocks to sell
        current_temp : Position
            Temporary position for simulation
        trade_start_time : pd.Timestamp
            Trading start time
        trade_end_time : pd.Timestamp
            Trading end time

        Returns
        -------
        list
            List of sell orders
        """
        sell_orders = []
        time_per_step = self.trade_calendar.get_freq()

        for stock_id in sell_stocks:
            if not self._is_stock_tradable_with_direction(stock_id, OrderDir.SELL, trade_start_time, trade_end_time):
                continue

            if current_temp.get_stock_count(stock_id, bar=time_per_step) < self.hold_thresh:
                continue

            sell_amount = current_temp.get_stock_amount(code=stock_id)
            sell_order = self._create_order(
                stock_id, sell_amount, OrderDir.SELL, trade_start_time, trade_end_time
            )

            if sell_order:
                sell_orders.append(sell_order)
                # Simulate trade to update cash
                self.trade_exchange.deal_order(sell_order, position=current_temp)

        return sell_orders

    def _generate_buy_orders(self, target_stocks, current_temp, trade_start_time, trade_end_time):
        """
        Generate buy orders to achieve equal weight target portfolio.

        Parameters
        ----------
        target_stocks : pd.Index
            Target stocks for portfolio
        current_temp : Position
            Temporary position with updated cash after sells
        trade_start_time : pd.Timestamp
            Trading start time
        trade_end_time : pd.Timestamp
            Trading end time

        Returns
        -------
        list
            List of buy orders
        """
        buy_orders = []
        total_value = current_temp.get_cash() + current_temp.calculate_stock_value()
        target_value_per_stock = total_value * self.risk_degree / self.topk if self.topk > 0 else 0

        for stock_id in target_stocks:
            current_amount = current_temp.get_stock_amount(stock_id)

            deal_price = self.trade_exchange.get_deal_price(
                stock_id=stock_id, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY
            )
            if deal_price is None:
                continue

            target_amount = self.trade_exchange.round_amount_by_trade_unit(
                target_value_per_stock / deal_price, DEFAULT_TRADE_UNIT_FACTOR
            )
            amount_diff = target_amount - current_amount

            if abs(amount_diff) > MIN_TRADE_AMOUNT_THRESHOLD:
                direction = Order.BUY if amount_diff > 0 else Order.SELL
                trade_direction = OrderDir.BUY if direction == Order.BUY else OrderDir.SELL

                if not self._is_stock_tradable_with_direction(stock_id, trade_direction, trade_start_time, trade_end_time):
                    continue

                buy_order = self._create_order(
                    stock_id, abs(amount_diff), direction, trade_start_time, trade_end_time
                )
                if buy_order:
                    buy_orders.append(buy_order)

        return buy_orders

    def generate_trade_decision(self, execute_result=None):
        """
        Generate trading decisions.

        This method executes the following steps:
        1. Get the latest prediction scores.
        2. Select Top-K or Bottom-K stocks based on scores and method_buy as target holdings.
        3. Sell stocks in current position that are not in the target list.
        4. Rebalance portfolio to achieve equal weight for target stocks.
        """
        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)

        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None or pred_score.empty:
            return TradeDecisionWO([], self)
        
        # Remove NaN values to avoid including them in target portfolio
        # This addresses the issue where NaN values could be selected when topk exceeds available valid scores
        pred_score = pred_score.dropna()
        if pred_score.empty:
            return TradeDecisionWO([], self)

        # 1. get topk stocks
        if self.method_buy == "top":
            sorted_score = pred_score.sort_values(ascending=False)
        elif self.method_buy == "bottom":
            sorted_score = pred_score.sort_values(ascending=True)
        else:
            raise NotImplementedError(f"method_buy '{self.method_buy}' is not supported!")

        target_stocks = self._get_tradable_stocks(sorted_score.index, trade_start_time, trade_end_time)

        # 2. Generate sell orders
        current_position = self.trade_position
        current_stock_list = current_position.get_stock_list()
        sell_stocks = pd.Index(current_stock_list).difference(target_stocks)

        # Create a temporary position to simulate trades and cash changes
        current_temp: Position = copy.deepcopy(current_position)
        sell_orders = self._generate_sell_orders(sell_stocks, current_temp, trade_start_time, trade_end_time)

        # 3. Generate buy orders for equal weight
        buy_orders = self._generate_buy_orders(target_stocks, current_temp, trade_start_time, trade_end_time)

        return TradeDecisionWO(sell_orders + buy_orders, self)


class WeightStrategyBase(BaseSignalStrategy):
    # TODO:
    # 1. Supporting leverage the get_range_limit result from the decision
    # 2. Supporting alter_outer_trade_decision
    # 3. Supporting checking the availability of trade decision
    def __init__(
        self,
        *,
        order_generator_cls_or_obj=OrderGenWOInteract,
        **kwargs,
    ):
        """
        signal :
            the information to describe a signal. Please refer to the docs of `qlib.backtest.signal.create_signal_from`
            the decision of the strategy will base on the given signal
        trade_exchange : Exchange
            exchange that provides market info, used to deal order and generate report

            - If `trade_exchange` is None, self.trade_exchange will be set with common_infra
            - It allowes different trade_exchanges is used in different executions.
            - For example:

                - In daily execution, both daily exchange and minutely are usable, but the daily exchange is recommended because it runs faster.
                - In minutely execution, the daily exchange is not usable, only the minutely exchange is recommended.
        """
        super().__init__(**kwargs)

        if isinstance(order_generator_cls_or_obj, type):
            self.order_generator: OrderGenerator = order_generator_cls_or_obj()
        else:
            self.order_generator: OrderGenerator = order_generator_cls_or_obj

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        """
        Generate target position from score for this date and the current position.The cash is not considered in the position

        Parameters
        -----------
        score : pd.Series
            pred score for this trade date, index is stock_id, contain 'score' column.
        current : Position()
            current position.
        trade_start_time: pd.Timestamp
        trade_end_time: pd.Timestamp
        """
        raise NotImplementedError()

    def generate_trade_decision(self, execute_result=None):
        # generate_trade_decision
        # generate_target_weight_position() and generate_order_list_from_target_weight_position() to generate order_list

        # get the number of trading step finished, trade_step can be [0, 1, 2, ..., trade_len - 1]
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if pred_score is None:
            return TradeDecisionWO([], self)
        current_temp = copy.deepcopy(self.trade_position)
        assert isinstance(current_temp, Position)  # Avoid InfPosition

        target_weight_position = self.generate_target_weight_position(
            score=pred_score, current=current_temp, trade_start_time=trade_start_time, trade_end_time=trade_end_time
        )
        order_list = self.order_generator.generate_order_list_from_target_weight_position(
            current=current_temp,
            trade_exchange=self.trade_exchange,
            risk_degree=self.get_risk_degree(trade_step),
            target_weight_position=target_weight_position,
            pred_start_time=pred_start_time,
            pred_end_time=pred_end_time,
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time,
        )
        return TradeDecisionWO(order_list, self)


class EnhancedIndexingStrategy(WeightStrategyBase):
    """Enhanced Indexing Strategy

    Enhanced indexing combines the arts of active management and passive management,
    with the aim of outperforming a benchmark index (e.g., S&P 500) in terms of
    portfolio return while controlling the risk exposure (a.k.a. tracking error).

    Users need to prepare their risk model data like below:

    .. code-block:: text

        ├── /path/to/riskmodel
        ├──── 20210101
        ├────── factor_exp.{csv|pkl|h5}
        ├────── factor_cov.{csv|pkl|h5}
        ├────── specific_risk.{csv|pkl|h5}
        ├────── blacklist.{csv|pkl|h5}  # optional

    The risk model data can be obtained from risk data provider. You can also use
    `qlib.model.riskmodel.structured.StructuredCovEstimator` to prepare these data.

    Args:
        riskmodel_path (str): risk model path
        name_mapping (dict): alternative file names
    """

    FACTOR_EXP_NAME = "factor_exp.pkl"
    FACTOR_COV_NAME = "factor_cov.pkl"
    SPECIFIC_RISK_NAME = "specific_risk.pkl"
    BLACKLIST_NAME = "blacklist.pkl"

    def __init__(
        self,
        *,
        riskmodel_root,
        market="csi500",
        turn_limit=None,
        name_mapping={},
        optimizer_kwargs={},
        verbose=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.logger = get_module_logger("EnhancedIndexingStrategy")

        self.riskmodel_root = riskmodel_root
        self.market = market
        self.turn_limit = turn_limit

        self.factor_exp_path = name_mapping.get("factor_exp", self.FACTOR_EXP_NAME)
        self.factor_cov_path = name_mapping.get("factor_cov", self.FACTOR_COV_NAME)
        self.specific_risk_path = name_mapping.get("specific_risk", self.SPECIFIC_RISK_NAME)
        self.blacklist_path = name_mapping.get("blacklist", self.BLACKLIST_NAME)

        self.optimizer = EnhancedIndexingOptimizer(**optimizer_kwargs)

        self.verbose = verbose

        self._riskdata_cache = {}

    def get_risk_data(self, date):
        if date in self._riskdata_cache:
            return self._riskdata_cache[date]

        root = self.riskmodel_root + "/" + date.strftime("%Y%m%d")
        if not os.path.exists(root):
            return None

        factor_exp = load_dataset(root + "/" + self.factor_exp_path, index_col=[0])
        factor_cov = load_dataset(root + "/" + self.factor_cov_path, index_col=[0])
        specific_risk = load_dataset(root + "/" + self.specific_risk_path, index_col=[0])

        if not factor_exp.index.equals(specific_risk.index):
            # NOTE: for stocks missing specific_risk, we always assume it has the highest volatility
            specific_risk = specific_risk.reindex(factor_exp.index, fill_value=specific_risk.max())

        universe = factor_exp.index.tolist()

        blacklist = []
        if os.path.exists(root + "/" + self.blacklist_path):
            blacklist = load_dataset(root + "/" + self.blacklist_path).index.tolist()

        self._riskdata_cache[date] = factor_exp.values, factor_cov.values, specific_risk.values, universe, blacklist

        return self._riskdata_cache[date]

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        trade_date = trade_start_time
        pre_date = get_pre_trading_date(trade_date, future=True)  # previous trade date

        # load risk data
        outs = self.get_risk_data(pre_date)
        if outs is None:
            self.logger.warning(f"no risk data for {pre_date:%Y-%m-%d}, skip optimization")
            return None
        factor_exp, factor_cov, specific_risk, universe, blacklist = outs

        # transform score
        # NOTE: for stocks missing score, we always assume they have the lowest score
        score = score.reindex(universe).fillna(score.min()).values

        # get current weight
        # NOTE: if a stock is not in universe, its current weight will be zero
        cur_weight = current.get_stock_weight_dict(only_stock=False)
        cur_weight = np.array([cur_weight.get(stock, 0) for stock in universe])
        assert all(cur_weight >= 0), "current weight has negative values"
        cur_weight = cur_weight / self.get_risk_degree(trade_date)  # sum of weight should be risk_degree
        if cur_weight.sum() > 1 and self.verbose:
            self.logger.warning(f"previous total holdings excess risk degree (current: {cur_weight.sum()})")

        # load bench weight
        bench_weight = D.features(
            D.instruments("all"), [f"${self.market}_weight"], start_time=pre_date, end_time=pre_date
        ).squeeze()
        bench_weight.index = bench_weight.index.droplevel(level="datetime")
        bench_weight = bench_weight.reindex(universe).fillna(0).values

        # whether stock tradable
        # NOTE: currently we use last day volume to check whether tradable
        tradable = D.features(D.instruments("all"), ["$volume"], start_time=pre_date, end_time=pre_date).squeeze()
        tradable.index = tradable.index.droplevel(level="datetime")
        tradable = tradable.reindex(universe).gt(0).values
        mask_force_hold = ~tradable

        # mask force sell
        mask_force_sell = np.array([stock in blacklist for stock in universe], dtype=bool)

        # optimize
        weight = self.optimizer(
            r=score,
            F=factor_exp,
            cov_b=factor_cov,
            var_u=specific_risk**2,
            w0=cur_weight,
            wb=bench_weight,
            mfh=mask_force_hold,
            mfs=mask_force_sell,
        )

        target_weight_position = {stock: weight for stock, weight in zip(universe, weight) if weight > 0}

        if self.verbose:
            self.logger.info("trade date: {:%Y-%m-%d}".format(trade_date))
            self.logger.info("number of holding stocks: {}".format(len(target_weight_position)))
            self.logger.info("total holding weight: {:.6f}".format(weight.sum()))

        return target_weight_position
