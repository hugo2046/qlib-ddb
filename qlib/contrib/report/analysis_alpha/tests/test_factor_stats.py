import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from qlib.contrib.report.analysis_alpha.factor_stats import FactorAnalyzer


def generate_mock_data(n_dates=100, n_stocks=50):
    """生成模拟数据"""
    dates = pd.date_range(end=datetime.now(), periods=n_dates, freq="B")
    stocks = [f"Stock_{i:03d}" for i in range(n_stocks)]

    # 模拟因子值 (随机)
    factor = pd.DataFrame(
        np.random.randn(n_dates, n_stocks), index=dates, columns=stocks
    )

    # 模拟收益率 (略微正相关于因子)
    ret_noise = np.random.randn(n_dates, n_stocks) * 0.02
    daily_ret = factor * 0.001 + ret_noise

    # 模拟交易滑点后的收益率 (略小于 daily_ret)
    trade_ret = daily_ret - 0.0002

    # 模拟涨跌停状态 (0:Normal, 1:UpLimit, -1:DownLimit)
    limit_vals = np.zeros((n_dates, n_stocks))
    mask = np.random.random((n_dates, n_stocks)) > 0.95
    # 生成随机的1或-1填充到mask位置
    random_limits = np.random.choice([1, -1], size=(n_dates, n_stocks))
    limit_vals[mask] = random_limits[mask]
    limit_status = pd.DataFrame(limit_vals, index=dates, columns=stocks)

    # 模拟停牌状态 (0:Normal, 1:Suspend)
    suspend_vals = np.zeros((n_dates, n_stocks))
    mask_sus = np.random.random((n_dates, n_stocks)) > 0.98
    suspend_vals[mask_sus] = 1
    suspend_status = pd.DataFrame(suspend_vals, index=dates, columns=stocks)

    return factor, daily_ret, trade_ret, limit_status, suspend_status


def test_factor_analyzer():
    print("Generating mock data...")
    factor, daily_ret, trade_ret, limit_status, suspend_status = generate_mock_data()

    print("\nInitializing FactorAnalyzer with Data...")
    analyzer = FactorAnalyzer(
        # Data
        factor=factor,
        daily_ret=daily_ret,
        trade_ret=trade_ret,
        limit_status=limit_status,
        suspend_status=suspend_status,
        # Config
        n_groups=5,
        booksize=1e7,
        cost=0.0003,
        filter_limit=True,
        filter_suspend=True,
        ic_method="spearman",
    )

    print("\nRunning Full Analysis...")
    pnl, detail, perf = analyzer.run_analysis(
        report_type="monthly",
    )

    print("\nPerformance Report:")
    analyzer.generate_report(perf)

    print("\nRunning Quantile Analysis...")
    q_results = analyzer.calc_quantile()

    for gid, res in q_results.items():
        print(f"Group {gid} Total Return: {res['ret'].sum():.4f}")


if __name__ == "__main__":
    try:
        test_factor_analyzer()
        print("\nTest Passed Successfully!")
    except Exception as e:
        print(f"\nTest Failed: {e}")
        import traceback

        traceback.print_exc()
