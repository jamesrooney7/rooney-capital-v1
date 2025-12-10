#!/usr/bin/env python3
"""
Comprehensive Portfolio Analysis for ML-Filtered Trading Strategies
Analyzes trade prediction files from ML meta-labeling optimization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration
TRADE_DATA_DIR = Path(__file__).parent / "trade_data"
INITIAL_CAPITAL = 250_000
RISK_FREE_RATE = 0.05  # 5% annual risk-free rate

def load_all_strategies():
    """Load all strategy trade files"""
    strategies = {}
    for file in TRADE_DATA_DIR.glob("*_held_out_predictions.csv"):
        name = file.stem.replace("_ml_meta_labeling_held_out_predictions", "")
        df = pd.read_csv(file, parse_dates=['Date'])
        df['strategy'] = name
        strategies[name] = df
    return strategies

def get_ml_filtered_trades(df):
    """Get only trades that pass ML filter (y_pred_binary=1)"""
    return df[df['y_pred_binary'] == 1].copy()

def calculate_portfolio_metrics(strategies, use_ml_filter=True):
    """Calculate comprehensive portfolio metrics"""

    # Combine all trades
    all_trades = []
    for name, df in strategies.items():
        if use_ml_filter:
            filtered = get_ml_filtered_trades(df)
        else:
            filtered = df.copy()
        all_trades.append(filtered)

    combined = pd.concat(all_trades, ignore_index=True)
    combined = combined.sort_values('Date').reset_index(drop=True)

    # Basic stats
    total_trades = len(combined)
    winning_trades = len(combined[combined['y_pnl_usd'] > 0])
    losing_trades = len(combined[combined['y_pnl_usd'] < 0])

    # P&L metrics
    total_pnl = combined['y_pnl_usd'].sum()
    gross_profit = combined[combined['y_pnl_usd'] > 0]['y_pnl_usd'].sum()
    gross_loss = abs(combined[combined['y_pnl_usd'] < 0]['y_pnl_usd'].sum())

    # Win rate and profit factor
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Average trade
    avg_win = combined[combined['y_pnl_usd'] > 0]['y_pnl_usd'].mean() if winning_trades > 0 else 0
    avg_loss = combined[combined['y_pnl_usd'] < 0]['y_pnl_usd'].mean() if losing_trades > 0 else 0
    avg_trade = combined['y_pnl_usd'].mean()

    # Calculate equity curve
    combined['cumulative_pnl'] = combined['y_pnl_usd'].cumsum()
    combined['equity'] = INITIAL_CAPITAL + combined['cumulative_pnl']

    # Drawdown calculation
    combined['peak'] = combined['equity'].cummax()
    combined['drawdown'] = combined['equity'] - combined['peak']
    combined['drawdown_pct'] = combined['drawdown'] / combined['peak'] * 100

    max_drawdown_usd = combined['drawdown'].min()
    max_drawdown_pct = combined['drawdown_pct'].min()

    # Date range
    start_date = combined['Date'].min()
    end_date = combined['Date'].max()
    trading_days = (end_date - start_date).days
    years = trading_days / 365.25

    # Annualized returns
    total_return_pct = (total_pnl / INITIAL_CAPITAL) * 100
    cagr = ((1 + total_pnl / INITIAL_CAPITAL) ** (1/years) - 1) * 100 if years > 0 else 0

    # Calculate daily returns for Sharpe/Sortino
    combined['date_only'] = combined['Date'].dt.date
    daily_pnl = combined.groupby('date_only')['y_pnl_usd'].sum()
    daily_returns = daily_pnl / INITIAL_CAPITAL

    # Sharpe Ratio (annualized)
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        excess_returns = daily_returns - RISK_FREE_RATE/252
        sharpe = np.sqrt(252) * excess_returns.mean() / daily_returns.std()
    else:
        sharpe = 0

    # Sortino Ratio (annualized)
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 1 and downside_returns.std() > 0:
        sortino = np.sqrt(252) * (daily_returns.mean() - RISK_FREE_RATE/252) / downside_returns.std()
    else:
        sortino = 0

    # Calmar Ratio
    calmar = cagr / abs(max_drawdown_pct) if max_drawdown_pct != 0 else 0

    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_trade': avg_trade,
        'max_drawdown_usd': max_drawdown_usd,
        'max_drawdown_pct': max_drawdown_pct,
        'total_return_pct': total_return_pct,
        'cagr': cagr,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'start_date': start_date,
        'end_date': end_date,
        'years': years,
        'combined_df': combined
    }

def analyze_trade_frequency(combined_df):
    """Analyze trade frequency patterns"""

    # Get date range
    start_date = combined_df['Date'].min().date()
    end_date = combined_df['Date'].max().date()

    # Create all trading days (weekdays)
    all_days = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days

    # Count trades per day
    combined_df['date_only'] = combined_df['Date'].dt.date
    trades_per_day = combined_df.groupby('date_only').size()

    # Create full date series
    full_series = pd.Series(0, index=[d.date() for d in all_days])
    for date, count in trades_per_day.items():
        if date in full_series.index:
            full_series[date] = count

    # Calculate stats
    total_trading_days = len(all_days)
    days_with_trades = (full_series > 0).sum()
    days_without_trades = (full_series == 0).sum()
    pct_zero_trade_days = days_without_trades / total_trading_days * 100

    avg_trades_per_day = full_series.mean()
    avg_trades_on_active_days = trades_per_day.mean()
    max_trades_per_day = full_series.max()

    # Consecutive zero-trade days
    zero_days = full_series == 0
    consecutive_zeros = []
    current_streak = 0
    for is_zero in zero_days:
        if is_zero:
            current_streak += 1
        else:
            if current_streak > 0:
                consecutive_zeros.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        consecutive_zeros.append(current_streak)

    max_consecutive_zero = max(consecutive_zeros) if consecutive_zeros else 0
    avg_consecutive_zero = np.mean(consecutive_zeros) if consecutive_zeros else 0

    return {
        'total_trading_days': total_trading_days,
        'days_with_trades': days_with_trades,
        'days_without_trades': days_without_trades,
        'pct_zero_trade_days': pct_zero_trade_days,
        'avg_trades_per_day': avg_trades_per_day,
        'avg_trades_on_active_days': avg_trades_on_active_days,
        'max_trades_per_day': max_trades_per_day,
        'max_consecutive_zero_days': max_consecutive_zero,
        'avg_consecutive_zero_days': avg_consecutive_zero,
        'trades_per_day_series': full_series
    }

def analyze_yearly_performance(combined_df):
    """Analyze performance by year"""
    combined_df['year'] = combined_df['Date'].dt.year

    yearly = combined_df.groupby('year').agg({
        'y_pnl_usd': ['sum', 'count', 'mean'],
        'y_true': 'mean'  # Win rate
    }).round(2)

    yearly.columns = ['total_pnl', 'num_trades', 'avg_trade', 'win_rate']
    yearly['win_rate'] = (yearly['win_rate'] * 100).round(1)
    yearly['return_pct'] = (yearly['total_pnl'] / INITIAL_CAPITAL * 100).round(2)

    return yearly

def analyze_monthly_performance(combined_df):
    """Analyze performance by month"""
    combined_df['year_month'] = combined_df['Date'].dt.to_period('M')

    monthly = combined_df.groupby('year_month').agg({
        'y_pnl_usd': ['sum', 'count', 'mean'],
        'y_true': 'mean'
    }).round(2)

    monthly.columns = ['total_pnl', 'num_trades', 'avg_trade', 'win_rate']
    monthly['win_rate'] = (monthly['win_rate'] * 100).round(1)
    monthly['return_pct'] = (monthly['total_pnl'] / INITIAL_CAPITAL * 100).round(2)

    return monthly

def analyze_strategy_correlation(strategies, use_ml_filter=True):
    """Calculate correlation between strategy daily returns"""

    # Get daily returns for each strategy
    daily_returns = {}

    for name, df in strategies.items():
        if use_ml_filter:
            filtered = get_ml_filtered_trades(df)
        else:
            filtered = df.copy()

        filtered['date_only'] = filtered['Date'].dt.date
        daily_pnl = filtered.groupby('date_only')['y_pnl_usd'].sum()
        daily_returns[name] = daily_pnl

    # Create DataFrame with all strategies
    returns_df = pd.DataFrame(daily_returns).fillna(0)

    # Calculate correlation matrix
    correlation = returns_df.corr()

    return correlation, returns_df

def analyze_per_strategy(strategies, use_ml_filter=True):
    """Get metrics for each individual strategy"""

    results = []
    for name, df in strategies.items():
        if use_ml_filter:
            filtered = get_ml_filtered_trades(df)
        else:
            filtered = df.copy()

        if len(filtered) == 0:
            continue

        total_pnl = filtered['y_pnl_usd'].sum()
        num_trades = len(filtered)
        win_rate = filtered['y_true'].mean() * 100
        avg_trade = filtered['y_pnl_usd'].mean()

        # Profit factor
        gross_profit = filtered[filtered['y_pnl_usd'] > 0]['y_pnl_usd'].sum()
        gross_loss = abs(filtered[filtered['y_pnl_usd'] < 0]['y_pnl_usd'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        results.append({
            'strategy': name,
            'num_trades': num_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_trade': avg_trade,
            'profit_factor': profit_factor,
            'trades_per_year': num_trades / 4  # ~4 years of data
        })

    return pd.DataFrame(results).sort_values('total_pnl', ascending=False)

def main():
    print("=" * 80)
    print("PORTFOLIO ANALYSIS - ML-FILTERED TRADING STRATEGIES")
    print("Held-Out Period: 2021-2024")
    print("=" * 80)

    # Load data
    strategies = load_all_strategies()
    print(f"\nLoaded {len(strategies)} strategies")

    # =========================================================================
    # PORTFOLIO METRICS (ML FILTERED)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PORTFOLIO METRICS (ML-FILTERED TRADES)")
    print("=" * 80)

    metrics = calculate_portfolio_metrics(strategies, use_ml_filter=True)

    print(f"\n{'Period:':<30} {metrics['start_date'].strftime('%Y-%m-%d')} to {metrics['end_date'].strftime('%Y-%m-%d')} ({metrics['years']:.1f} years)")
    print(f"{'Initial Capital:':<30} ${INITIAL_CAPITAL:,.0f}")

    print(f"\n--- TRADE STATISTICS ---")
    print(f"{'Total Trades:':<30} {metrics['total_trades']:,}")
    print(f"{'Winning Trades:':<30} {metrics['winning_trades']:,}")
    print(f"{'Losing Trades:':<30} {metrics['losing_trades']:,}")
    print(f"{'Win Rate:':<30} {metrics['win_rate']:.1f}%")

    print(f"\n--- P&L METRICS ---")
    print(f"{'Total P&L:':<30} ${metrics['total_pnl']:,.2f}")
    print(f"{'Gross Profit:':<30} ${metrics['gross_profit']:,.2f}")
    print(f"{'Gross Loss:':<30} ${metrics['gross_loss']:,.2f}")
    print(f"{'Profit Factor:':<30} {metrics['profit_factor']:.2f}")
    print(f"{'Average Win:':<30} ${metrics['avg_win']:,.2f}")
    print(f"{'Average Loss:':<30} ${metrics['avg_loss']:,.2f}")
    print(f"{'Average Trade:':<30} ${metrics['avg_trade']:,.2f}")

    print(f"\n--- RISK METRICS ---")
    print(f"{'Max Drawdown ($):':<30} ${metrics['max_drawdown_usd']:,.2f}")
    print(f"{'Max Drawdown (%):':<30} {metrics['max_drawdown_pct']:.2f}%")

    print(f"\n--- RETURN METRICS ---")
    print(f"{'Total Return:':<30} {metrics['total_return_pct']:.2f}%")
    print(f"{'CAGR:':<30} {metrics['cagr']:.2f}%")
    print(f"{'Sharpe Ratio:':<30} {metrics['sharpe']:.2f}")
    print(f"{'Sortino Ratio:':<30} {metrics['sortino']:.2f}")
    print(f"{'Calmar Ratio:':<30} {metrics['calmar']:.2f}")

    # =========================================================================
    # TRADE FREQUENCY ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("TRADE FREQUENCY ANALYSIS")
    print("=" * 80)

    freq = analyze_trade_frequency(metrics['combined_df'])

    print(f"\n{'Total Trading Days:':<35} {freq['total_trading_days']:,}")
    print(f"{'Days WITH Trades:':<35} {freq['days_with_trades']:,}")
    print(f"{'Days WITHOUT Trades:':<35} {freq['days_without_trades']:,}")
    print(f"{'% Zero-Trade Days:':<35} {freq['pct_zero_trade_days']:.1f}%")
    print(f"\n{'Avg Trades/Day (all days):':<35} {freq['avg_trades_per_day']:.2f}")
    print(f"{'Avg Trades/Day (active days):':<35} {freq['avg_trades_on_active_days']:.2f}")
    print(f"{'Max Trades in One Day:':<35} {freq['max_trades_per_day']}")
    print(f"\n{'Max Consecutive Zero-Trade Days:':<35} {freq['max_consecutive_zero_days']}")
    print(f"{'Avg Consecutive Zero-Trade Days:':<35} {freq['avg_consecutive_zero_days']:.1f}")

    # Distribution of trades per day
    print(f"\n--- TRADES PER DAY DISTRIBUTION ---")
    tpd = freq['trades_per_day_series']
    for i in range(min(6, int(tpd.max()) + 1)):
        count = (tpd == i).sum()
        pct = count / len(tpd) * 100
        print(f"  {i} trades/day: {count:4d} days ({pct:5.1f}%)")
    if tpd.max() >= 6:
        count = (tpd >= 6).sum()
        pct = count / len(tpd) * 100
        print(f"  6+ trades/day: {count:4d} days ({pct:5.1f}%)")

    # =========================================================================
    # YEARLY PERFORMANCE
    # =========================================================================
    print("\n" + "=" * 80)
    print("YEARLY PERFORMANCE")
    print("=" * 80)

    yearly = analyze_yearly_performance(metrics['combined_df'])
    print(f"\n{'Year':<8} {'P&L':>12} {'Return%':>10} {'Trades':>8} {'Win%':>8} {'Avg Trade':>12}")
    print("-" * 60)
    for year, row in yearly.iterrows():
        print(f"{year:<8} ${row['total_pnl']:>10,.0f} {row['return_pct']:>9.1f}% {int(row['num_trades']):>8} {row['win_rate']:>7.1f}% ${row['avg_trade']:>10,.0f}")

    # =========================================================================
    # MONTHLY PERFORMANCE
    # =========================================================================
    print("\n" + "=" * 80)
    print("MONTHLY PERFORMANCE")
    print("=" * 80)

    monthly = analyze_monthly_performance(metrics['combined_df'])

    # Show monthly returns in a grid format
    print(f"\n{'Month':<10}", end="")
    for year in sorted(monthly.index.year.unique()):
        print(f"{year:>10}", end="")
    print()
    print("-" * 55)

    for month in range(1, 13):
        month_name = datetime(2000, month, 1).strftime('%b')
        print(f"{month_name:<10}", end="")
        for year in sorted(monthly.index.year.unique()):
            try:
                period = pd.Period(f"{year}-{month:02d}")
                if period in monthly.index:
                    pnl = monthly.loc[period, 'return_pct']
                    print(f"{pnl:>9.1f}%", end="")
                else:
                    print(f"{'---':>10}", end="")
            except:
                print(f"{'---':>10}", end="")
        print()

    # Monthly statistics
    print(f"\n--- MONTHLY STATISTICS ---")
    positive_months = (monthly['total_pnl'] > 0).sum()
    negative_months = (monthly['total_pnl'] <= 0).sum()
    total_months = len(monthly)
    print(f"{'Positive Months:':<25} {positive_months} ({positive_months/total_months*100:.1f}%)")
    print(f"{'Negative Months:':<25} {negative_months} ({negative_months/total_months*100:.1f}%)")
    print(f"{'Best Month:':<25} ${monthly['total_pnl'].max():,.0f} ({monthly['total_pnl'].idxmax()})")
    print(f"{'Worst Month:':<25} ${monthly['total_pnl'].min():,.0f} ({monthly['total_pnl'].idxmin()})")
    print(f"{'Avg Monthly P&L:':<25} ${monthly['total_pnl'].mean():,.0f}")
    print(f"{'Avg Monthly Trades:':<25} {monthly['num_trades'].mean():.1f}")

    # =========================================================================
    # PER-STRATEGY BREAKDOWN
    # =========================================================================
    print("\n" + "=" * 80)
    print("PER-STRATEGY BREAKDOWN (ML-FILTERED)")
    print("=" * 80)

    strategy_stats = analyze_per_strategy(strategies, use_ml_filter=True)
    print(f"\n{'Strategy':<35} {'Trades':>8} {'P&L':>12} {'Win%':>8} {'PF':>8} {'Trades/Yr':>10}")
    print("-" * 85)
    for _, row in strategy_stats.iterrows():
        pf = f"{row['profit_factor']:.2f}" if row['profit_factor'] != float('inf') else "inf"
        print(f"{row['strategy']:<35} {int(row['num_trades']):>8} ${row['total_pnl']:>10,.0f} {row['win_rate']:>7.1f}% {pf:>8} {row['trades_per_year']:>10.1f}")

    # =========================================================================
    # STRATEGY CORRELATION
    # =========================================================================
    print("\n" + "=" * 80)
    print("STRATEGY CORRELATION (Daily Returns)")
    print("=" * 80)

    correlation, returns_df = analyze_strategy_correlation(strategies, use_ml_filter=True)

    # Simplify names for display
    short_names = {name: name.split('_')[0] + '_' + name.split('_')[1][:3] for name in correlation.columns}

    print("\nCorrelation Matrix (showing values > 0.3 or < -0.3):")
    print()

    # Print header
    print(f"{'':>15}", end="")
    for col in correlation.columns:
        print(f"{short_names[col]:>12}", end="")
    print()

    for row_name in correlation.index:
        print(f"{short_names[row_name]:>15}", end="")
        for col_name in correlation.columns:
            val = correlation.loc[row_name, col_name]
            if row_name == col_name:
                print(f"{'1.00':>12}", end="")
            elif abs(val) >= 0.3:
                print(f"{val:>12.2f}", end="")
            else:
                print(f"{'·':>12}", end="")
        print()

    # Average correlation
    upper_tri = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
    avg_corr = upper_tri.stack().mean()
    print(f"\nAverage Pairwise Correlation: {avg_corr:.3f}")

    # =========================================================================
    # PROBABILITY OF ZERO-TRADE PERIODS
    # =========================================================================
    print("\n" + "=" * 80)
    print("PROBABILITY OF ZERO-TRADE PERIODS")
    print("=" * 80)

    tpd = freq['trades_per_day_series']
    zero_days = tpd == 0

    # Calculate probability of N consecutive zero days
    print(f"\nBased on historical data ({freq['pct_zero_trade_days']:.1f}% zero-trade days):")
    print()

    p_zero = freq['pct_zero_trade_days'] / 100

    print(f"{'Consecutive Days':<20} {'Probability':>15} {'Expected/Year':>15}")
    print("-" * 50)
    for n in [1, 2, 3, 4, 5, 7, 10]:
        # Simplified probability (assuming independence)
        prob = p_zero ** n
        expected_per_year = 252 * prob  # rough estimate
        print(f"{n:<20} {prob*100:>14.2f}% {expected_per_year:>15.1f}")

    # Actual consecutive zero periods from data
    print(f"\n--- ACTUAL CONSECUTIVE ZERO-TRADE PERIODS ---")
    zero_runs = []
    current_run = 0
    for is_zero in zero_days:
        if is_zero:
            current_run += 1
        else:
            if current_run > 0:
                zero_runs.append(current_run)
            current_run = 0
    if current_run > 0:
        zero_runs.append(current_run)

    if zero_runs:
        zero_runs_series = pd.Series(zero_runs)
        print(f"Max consecutive zero days: {zero_runs_series.max()}")
        print(f"Mean consecutive zero days: {zero_runs_series.mean():.1f}")
        print(f"Number of zero-day streaks: {len(zero_runs)}")
        print(f"\nDistribution of streak lengths:")
        for length in sorted(set(zero_runs)):
            count = zero_runs.count(length)
            print(f"  {length} day(s): {count} occurrences")

    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)

if __name__ == "__main__":
    main()
