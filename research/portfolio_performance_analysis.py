#!/usr/bin/env python3
"""
Comprehensive Portfolio Performance Analysis

Calculates institutional-grade performance metrics for portfolio optimization results.
This script provides the deep analysis that professional algo funds use to evaluate strategies.

Usage:
    python research/portfolio_performance_analysis.py \
        --results-dir results \
        --max-positions 4 \
        --daily-stop-loss 2500 \
        --output-dir analysis

Metrics Calculated:
    - Returns: Total, CAGR, monthly/yearly breakdowns
    - Risk: Sharpe, Sortino, Calmar, volatility, downside deviation
    - Drawdowns: Max DD, avg DD, recovery time, underwater periods
    - Win/Loss: Win rate, avg win/loss, profit factor, expectancy
    - Distribution: Skewness, kurtosis, tail risk (VaR, CVaR)
    - Consistency: Monthly win rate, rolling Sharpe, up/down capture
    - Risk-Adjusted: Information ratio, Omega ratio, gain-to-pain
    - Correlation: To benchmark, monthly correlation stability
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from scipy import stats

# Import the simulator
import sys
sys.path.insert(0, str(Path(__file__).parent))
from portfolio_simulator import (
    discover_available_symbols,
    load_symbol_trades,
    simulate_portfolio_intraday
)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PortfolioPerformanceAnalyzer:
    """Comprehensive performance analysis for portfolio strategies."""

    def __init__(self, equity_df: pd.DataFrame, initial_capital: float = 100000):
        """
        Initialize analyzer with equity curve.

        Args:
            equity_df: DataFrame with columns ['time', 'equity', 'n_positions', 'daily_pnl', 'stopped_out']
            initial_capital: Starting capital
        """
        self.equity_df = equity_df.copy()
        self.initial_capital = initial_capital

        # Ensure time is datetime
        self.equity_df['time'] = pd.to_datetime(self.equity_df['time'])
        self.equity_df = self.equity_df.sort_values('time').reset_index(drop=True)

        # Create daily resampled equity for many calculations
        self.daily_equity = self.equity_df.set_index('time')['equity'].resample('D').last().ffill()
        self.daily_returns = self.daily_equity.pct_change().dropna()

        # Time span
        self.start_date = self.equity_df['time'].iloc[0]
        self.end_date = self.equity_df['time'].iloc[-1]
        self.days = (self.end_date - self.start_date).days
        self.years = self.days / 365.25

    def calculate_all_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        metrics = {}

        # Basic returns
        metrics.update(self._calculate_returns())

        # Risk metrics
        metrics.update(self._calculate_risk_metrics())

        # Drawdown analysis
        metrics.update(self._calculate_drawdown_metrics())

        # Win/loss statistics
        metrics.update(self._calculate_win_loss_metrics())

        # Distribution analysis
        metrics.update(self._calculate_distribution_metrics())

        # Risk-adjusted returns
        metrics.update(self._calculate_risk_adjusted_metrics())

        # Consistency metrics
        metrics.update(self._calculate_consistency_metrics())

        # Monthly/Yearly breakdowns
        metrics.update(self._calculate_period_breakdowns())

        return metrics

    def _calculate_returns(self) -> Dict:
        """Calculate return metrics."""
        final_equity = self.equity_df['equity'].iloc[-1]
        total_return = final_equity - self.initial_capital
        total_return_pct = total_return / self.initial_capital

        # CAGR
        cagr = (1 + total_return_pct) ** (1 / self.years) - 1 if self.years > 0 else 0

        # Average daily return
        avg_daily_return = self.daily_returns.mean()
        avg_daily_return_annualized = avg_daily_return * 252

        return {
            'total_return_dollars': total_return,
            'total_return_pct': total_return_pct * 100,
            'cagr_pct': cagr * 100,
            'avg_daily_return_pct': avg_daily_return * 100,
            'avg_monthly_return_pct': (avg_daily_return * 21) * 100,
            'avg_yearly_return_pct': avg_daily_return_annualized * 100,
        }

    def _calculate_risk_metrics(self) -> Dict:
        """Calculate risk and volatility metrics."""
        # Volatility
        daily_vol = self.daily_returns.std()
        annual_vol = daily_vol * np.sqrt(252)

        # Downside deviation (semi-deviation)
        negative_returns = self.daily_returns[self.daily_returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252)

        # Sharpe ratio (assuming 0% risk-free rate for futures)
        cagr = (1 + (self.equity_df['equity'].iloc[-1] / self.initial_capital - 1)) ** (1 / self.years) - 1
        sharpe = cagr / annual_vol if annual_vol > 0 else 0

        # Sortino ratio (uses downside deviation instead of total volatility)
        sortino = cagr / downside_deviation if downside_deviation > 0 else 0

        return {
            'volatility_daily_pct': daily_vol * 100,
            'volatility_annual_pct': annual_vol * 100,
            'downside_deviation_annual_pct': downside_deviation * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
        }

    def _calculate_drawdown_metrics(self) -> Dict:
        """Calculate drawdown metrics."""
        equity = self.daily_equity

        # Running maximum
        running_max = equity.expanding().max()

        # Drawdown in dollars and percent
        drawdown_dollars = equity - running_max
        drawdown_pct = (equity - running_max) / running_max

        # Max drawdown
        max_dd_dollars = drawdown_dollars.min()
        max_dd_pct = drawdown_pct.min()

        # Calmar ratio (CAGR / Max DD %)
        cagr = (1 + (self.equity_df['equity'].iloc[-1] / self.initial_capital - 1)) ** (1 / self.years) - 1
        calmar = abs(cagr / max_dd_pct) if max_dd_pct != 0 else 0

        # Average drawdown
        avg_dd_dollars = drawdown_dollars[drawdown_dollars < 0].mean() if (drawdown_dollars < 0).any() else 0
        avg_dd_pct = drawdown_pct[drawdown_pct < 0].mean() if (drawdown_pct < 0).any() else 0

        # Underwater periods (time spent in drawdown)
        underwater = drawdown_dollars < 0
        pct_time_underwater = (underwater.sum() / len(underwater)) * 100

        # Recovery analysis
        recovery_times = []
        in_drawdown = False
        drawdown_start = None

        for date, dd in drawdown_dollars.items():
            if dd < 0 and not in_drawdown:
                # Starting new drawdown
                in_drawdown = True
                drawdown_start = date
            elif dd >= 0 and in_drawdown:
                # Recovered from drawdown
                in_drawdown = False
                if drawdown_start:
                    recovery_days = (date - drawdown_start).days
                    recovery_times.append(recovery_days)

        avg_recovery_days = np.mean(recovery_times) if recovery_times else 0
        max_recovery_days = np.max(recovery_times) if recovery_times else 0

        # Max drawdown duration (longest period underwater)
        max_dd_duration = 0
        current_duration = 0
        for is_underwater in underwater:
            if is_underwater:
                current_duration += 1
                max_dd_duration = max(max_dd_duration, current_duration)
            else:
                current_duration = 0

        return {
            'max_drawdown_dollars': max_dd_dollars,
            'max_drawdown_pct': max_dd_pct * 100,
            'avg_drawdown_dollars': avg_dd_dollars,
            'avg_drawdown_pct': avg_dd_pct * 100,
            'calmar_ratio': calmar,
            'pct_time_underwater': pct_time_underwater,
            'avg_recovery_days': avg_recovery_days,
            'max_recovery_days': max_recovery_days,
            'max_drawdown_duration_days': max_dd_duration,
        }

    def _calculate_win_loss_metrics(self) -> Dict:
        """Calculate win/loss statistics."""
        # Daily P&L changes
        daily_pnl = self.daily_equity.diff().dropna()

        # Wins and losses
        wins = daily_pnl[daily_pnl > 0]
        losses = daily_pnl[daily_pnl < 0]

        n_wins = len(wins)
        n_losses = len(losses)
        n_total = len(daily_pnl)

        win_rate = (n_wins / n_total * 100) if n_total > 0 else 0

        # Average win/loss
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0

        # Win/loss ratio
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        # Profit factor
        gross_profit = wins.sum()
        gross_loss = abs(losses.sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        # Expectancy (average $ per day)
        expectancy = daily_pnl.mean()

        # Largest win/loss
        largest_win = wins.max() if len(wins) > 0 else 0
        largest_loss = abs(losses.min()) if len(losses) > 0 else 0

        return {
            'win_rate_pct': win_rate,
            'avg_win_dollars': avg_win,
            'avg_loss_dollars': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'profit_factor': profit_factor,
            'expectancy_daily_dollars': expectancy,
            'largest_win_dollars': largest_win,
            'largest_loss_dollars': largest_loss,
            'total_winning_days': n_wins,
            'total_losing_days': n_losses,
        }

    def _calculate_distribution_metrics(self) -> Dict:
        """Calculate distribution and tail risk metrics."""
        returns = self.daily_returns

        # Skewness (asymmetry of return distribution)
        skewness = returns.skew()

        # Kurtosis (tail heaviness)
        kurtosis = returns.kurtosis()

        # Value at Risk (VaR) - 95% and 99%
        var_95 = returns.quantile(0.05) * 100  # 5th percentile (95% VaR)
        var_99 = returns.quantile(0.01) * 100  # 1st percentile (99% VaR)

        # Conditional Value at Risk (CVaR / Expected Shortfall)
        # Average of returns worse than VaR threshold
        cvar_95 = returns[returns <= returns.quantile(0.05)].mean() * 100
        cvar_99 = returns[returns <= returns.quantile(0.01)].mean() * 100

        # Worst day
        worst_day = returns.min() * 100
        best_day = returns.max() * 100

        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_95_pct': var_95,
            'var_99_pct': var_99,
            'cvar_95_pct': cvar_95,
            'cvar_99_pct': cvar_99,
            'worst_day_pct': worst_day,
            'best_day_pct': best_day,
        }

    def _calculate_risk_adjusted_metrics(self) -> Dict:
        """Calculate advanced risk-adjusted return metrics."""
        returns = self.daily_returns
        cagr = (1 + (self.equity_df['equity'].iloc[-1] / self.initial_capital - 1)) ** (1 / self.years) - 1

        # Omega Ratio (probability-weighted ratio of gains vs losses)
        # Threshold = 0 (no risk-free rate for futures)
        threshold = 0
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]

        omega_ratio = gains.sum() / losses.sum() if losses.sum() > 0 else np.inf

        # Gain-to-Pain Ratio (sum of returns / sum of absolute drawdowns)
        cumulative_return = (self.daily_equity / self.initial_capital - 1).sum()
        drawdowns = self.daily_equity - self.daily_equity.expanding().max()
        sum_pain = abs(drawdowns[drawdowns < 0].sum())

        gain_to_pain = (self.daily_equity.iloc[-1] - self.initial_capital) / sum_pain if sum_pain > 0 else 0

        # Ulcer Index (measure of downside volatility)
        running_max = self.daily_equity.expanding().max()
        drawdown_pct = ((self.daily_equity - running_max) / running_max) * 100
        ulcer_index = np.sqrt((drawdown_pct ** 2).mean())

        # Ulcer Performance Index (UPI) - similar to Sharpe but uses Ulcer Index
        upi = (cagr * 100) / ulcer_index if ulcer_index > 0 else 0

        return {
            'omega_ratio': omega_ratio,
            'gain_to_pain_ratio': gain_to_pain,
            'ulcer_index': ulcer_index,
            'ulcer_performance_index': upi,
        }

    def _calculate_consistency_metrics(self) -> Dict:
        """Calculate consistency and stability metrics."""
        # Monthly returns
        monthly_equity = self.daily_equity.resample('M').last()
        monthly_returns = monthly_equity.pct_change().dropna()

        # Monthly win rate
        monthly_wins = (monthly_returns > 0).sum()
        monthly_total = len(monthly_returns)
        monthly_win_rate = (monthly_wins / monthly_total * 100) if monthly_total > 0 else 0

        # Rolling Sharpe (6-month windows)
        rolling_sharpe = []
        for i in range(126, len(self.daily_returns)):  # 126 trading days ≈ 6 months
            window_returns = self.daily_returns.iloc[i-126:i]
            window_sharpe = (window_returns.mean() * 252) / (window_returns.std() * np.sqrt(252))
            rolling_sharpe.append(window_sharpe)

        avg_rolling_sharpe = np.mean(rolling_sharpe) if rolling_sharpe else 0
        std_rolling_sharpe = np.std(rolling_sharpe) if rolling_sharpe else 0

        # Consistency score (% of positive months)
        positive_months = (monthly_returns > 0).sum()
        consistency_score = (positive_months / len(monthly_returns) * 100) if len(monthly_returns) > 0 else 0

        return {
            'monthly_win_rate_pct': monthly_win_rate,
            'positive_months': int(positive_months),
            'total_months': int(monthly_total),
            'avg_rolling_sharpe_6m': avg_rolling_sharpe,
            'std_rolling_sharpe_6m': std_rolling_sharpe,
            'consistency_score_pct': consistency_score,
        }

    def _calculate_period_breakdowns(self) -> Dict:
        """Calculate monthly and yearly return breakdowns."""
        # Monthly breakdown
        monthly_equity = self.daily_equity.resample('M').last()
        monthly_returns = monthly_equity.pct_change().dropna()

        monthly_breakdown = []
        for date, ret in monthly_returns.items():
            monthly_breakdown.append({
                'year': date.year,
                'month': date.month,
                'month_name': date.strftime('%b'),
                'return_pct': ret * 100,
                'equity': monthly_equity.loc[date]
            })

        # Yearly breakdown
        yearly_equity = self.daily_equity.resample('Y').last()
        yearly_returns = yearly_equity.pct_change().dropna()

        yearly_breakdown = []
        for date, ret in yearly_returns.items():
            yearly_breakdown.append({
                'year': date.year,
                'return_pct': ret * 100,
                'equity': yearly_equity.loc[date]
            })

        # Add first year separately if data starts mid-year
        first_year = self.start_date.year
        if first_year not in [y['year'] for y in yearly_breakdown]:
            first_year_equity = self.daily_equity[self.daily_equity.index.year == first_year]
            if len(first_year_equity) > 0:
                first_year_return = (first_year_equity.iloc[-1] / self.initial_capital - 1) * 100
                yearly_breakdown.insert(0, {
                    'year': first_year,
                    'return_pct': first_year_return,
                    'equity': first_year_equity.iloc[-1]
                })

        return {
            'monthly_returns': monthly_breakdown,
            'yearly_returns': yearly_breakdown,
        }

    def generate_report(self, metrics: Dict, output_file: Optional[Path] = None) -> str:
        """Generate formatted performance report."""
        report_lines = [
            "=" * 80,
            "COMPREHENSIVE PORTFOLIO PERFORMANCE ANALYSIS",
            "=" * 80,
            "",
            f"Analysis Period: {self.start_date.date()} to {self.end_date.date()}",
            f"Duration: {self.days} days ({self.years:.2f} years)",
            f"Initial Capital: ${self.initial_capital:,.0f}",
            "",
            "=" * 80,
            "RETURNS",
            "=" * 80,
            f"Total Return:          ${metrics['total_return_dollars']:>12,.2f}  ({metrics['total_return_pct']:>6.2f}%)",
            f"CAGR:                  {metrics['cagr_pct']:>6.2f}%",
            f"Avg Daily Return:      {metrics['avg_daily_return_pct']:>6.2f}%",
            f"Avg Monthly Return:    {metrics['avg_monthly_return_pct']:>6.2f}%",
            f"Avg Yearly Return:     {metrics['avg_yearly_return_pct']:>6.2f}%",
            "",
            "=" * 80,
            "RISK METRICS",
            "=" * 80,
            f"Sharpe Ratio:          {metrics['sharpe_ratio']:>12.2f}",
            f"Sortino Ratio:         {metrics['sortino_ratio']:>12.2f}",
            f"Calmar Ratio:          {metrics['calmar_ratio']:>12.2f}",
            f"Annual Volatility:     {metrics['volatility_annual_pct']:>6.2f}%",
            f"Downside Deviation:    {metrics['downside_deviation_annual_pct']:>6.2f}%",
            "",
            "=" * 80,
            "DRAWDOWN ANALYSIS",
            "=" * 80,
            f"Max Drawdown:          ${metrics['max_drawdown_dollars']:>12,.2f}  ({metrics['max_drawdown_pct']:>6.2f}%)",
            f"Avg Drawdown:          ${metrics['avg_drawdown_dollars']:>12,.2f}  ({metrics['avg_drawdown_pct']:>6.2f}%)",
            f"Time Underwater:       {metrics['pct_time_underwater']:>6.2f}%",
            f"Avg Recovery Time:     {metrics['avg_recovery_days']:>12.0f} days",
            f"Max Recovery Time:     {metrics['max_recovery_days']:>12.0f} days",
            f"Max DD Duration:       {metrics['max_drawdown_duration_days']:>12.0f} days",
            "",
            "=" * 80,
            "WIN/LOSS STATISTICS",
            "=" * 80,
            f"Win Rate:              {metrics['win_rate_pct']:>6.2f}%",
            f"Profit Factor:         {metrics['profit_factor']:>12.2f}",
            f"Win/Loss Ratio:        {metrics['win_loss_ratio']:>12.2f}",
            f"Avg Win:               ${metrics['avg_win_dollars']:>12,.2f}",
            f"Avg Loss:              ${metrics['avg_loss_dollars']:>12,.2f}",
            f"Largest Win:           ${metrics['largest_win_dollars']:>12,.2f}",
            f"Largest Loss:          ${metrics['largest_loss_dollars']:>12,.2f}",
            f"Daily Expectancy:      ${metrics['expectancy_daily_dollars']:>12,.2f}",
            f"Winning Days:          {metrics['total_winning_days']:>12}",
            f"Losing Days:           {metrics['total_losing_days']:>12}",
            "",
            "=" * 80,
            "DISTRIBUTION & TAIL RISK",
            "=" * 80,
            f"Skewness:              {metrics['skewness']:>12.3f}",
            f"Kurtosis:              {metrics['kurtosis']:>12.3f}",
            f"VaR (95%):             {metrics['var_95_pct']:>6.2f}%  (worst 5% of days)",
            f"VaR (99%):             {metrics['var_99_pct']:>6.2f}%  (worst 1% of days)",
            f"CVaR (95%):            {metrics['cvar_95_pct']:>6.2f}%  (avg of worst 5%)",
            f"CVaR (99%):            {metrics['cvar_99_pct']:>6.2f}%  (avg of worst 1%)",
            f"Best Day:              {metrics['best_day_pct']:>6.2f}%",
            f"Worst Day:             {metrics['worst_day_pct']:>6.2f}%",
            "",
            "=" * 80,
            "RISK-ADJUSTED PERFORMANCE",
            "=" * 80,
            f"Omega Ratio:           {metrics['omega_ratio']:>12.2f}",
            f"Gain-to-Pain:          {metrics['gain_to_pain_ratio']:>12.2f}",
            f"Ulcer Index:           {metrics['ulcer_index']:>12.2f}",
            f"Ulcer Performance Index: {metrics['ulcer_performance_index']:>12.2f}",
            "",
            "=" * 80,
            "CONSISTENCY METRICS",
            "=" * 80,
            f"Monthly Win Rate:      {metrics['monthly_win_rate_pct']:>6.2f}%",
            f"Positive Months:       {metrics['positive_months']:>12} / {metrics['total_months']}",
            f"Consistency Score:     {metrics['consistency_score_pct']:>6.2f}%",
            f"Avg 6M Rolling Sharpe: {metrics['avg_rolling_sharpe_6m']:>12.2f}",
            f"Std 6M Rolling Sharpe: {metrics['std_rolling_sharpe_6m']:>12.2f}",
            "",
            "=" * 80,
            "YEARLY RETURNS",
            "=" * 80,
        ]

        for yr in metrics['yearly_returns']:
            report_lines.append(f"{yr['year']}:  {yr['return_pct']:>8.2f}%    Equity: ${yr['equity']:>12,.2f}")

        report_lines.extend([
            "",
            "=" * 80,
            "MONTHLY RETURNS",
            "=" * 80,
            ""
        ])

        # Group by year for monthly display
        monthly_by_year = {}
        for m in metrics['monthly_returns']:
            year = m['year']
            if year not in monthly_by_year:
                monthly_by_year[year] = []
            monthly_by_year[year].append(m)

        for year in sorted(monthly_by_year.keys()):
            report_lines.append(f"{year}:")
            for m in monthly_by_year[year]:
                sign = "+" if m['return_pct'] >= 0 else ""
                report_lines.append(f"  {m['month_name']:>3}: {sign}{m['return_pct']:>7.2f}%")
            report_lines.append("")

        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        if output_file:
            output_file.write_text(report)
            logger.info(f"Report saved to: {output_file}")

        return report


def main():
    parser = argparse.ArgumentParser(description='Comprehensive portfolio performance analysis')
    parser.add_argument('--results-dir', type=Path, required=True,
                      help='Directory containing optimization results')
    parser.add_argument('--max-positions', type=int, required=True,
                      help='Maximum concurrent positions')
    parser.add_argument('--daily-stop-loss', type=float, default=2500,
                      help='Daily stop loss in dollars (default: 2500)')
    parser.add_argument('--output-dir', type=Path, default=Path('analysis'),
                      help='Output directory for analysis reports')
    parser.add_argument('--symbols', nargs='+', default=None,
                      help='Symbols to include (default: all available)')

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("COMPREHENSIVE PORTFOLIO PERFORMANCE ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Max positions: {args.max_positions}")
    logger.info(f"Daily stop loss: ${args.daily_stop_loss:,.0f}")

    # Discover symbols
    available_symbols = discover_available_symbols(args.results_dir)

    if args.symbols:
        symbols = [s for s in args.symbols if s in available_symbols]
        logger.info(f"Using specified symbols: {symbols}")
    else:
        symbols = available_symbols
        logger.info(f"Using all available symbols: {symbols}")

    if not symbols:
        logger.error("No symbols found with trade data")
        return

    # Load trades for all symbols
    logger.info("\nLoading trade data...")
    symbol_trades = {}
    symbol_metadata = {}
    for symbol in symbols:
        try:
            trades_df, metadata = load_symbol_trades(args.results_dir, symbol)
            symbol_trades[symbol] = trades_df
            symbol_metadata[symbol] = metadata
            logger.info(f"  {symbol}: {len(trades_df)} trades")
        except Exception as e:
            logger.warning(f"  {symbol}: Failed to load - {e}")

    if not symbol_trades:
        logger.error("No trade data loaded")
        return

    # Run portfolio simulation
    logger.info(f"\nRunning portfolio simulation (max_positions={args.max_positions})...")
    equity_df, sim_metrics = simulate_portfolio_intraday(
        symbol_trades,
        symbol_metadata,
        max_positions=args.max_positions,
        daily_stop_loss=args.daily_stop_loss,
        initial_capital=100000
    )

    if equity_df.empty:
        logger.error("Simulation produced no results")
        return

    # Run comprehensive analysis
    logger.info("\nCalculating comprehensive performance metrics...")
    analyzer = PortfolioPerformanceAnalyzer(equity_df, initial_capital=100000)
    metrics = analyzer.calculate_all_metrics()

    # Generate report
    logger.info("\nGenerating performance report...")
    report_file = args.output_dir / f"portfolio_analysis_max{args.max_positions}.txt"
    report = analyzer.generate_report(metrics, output_file=report_file)

    # Print to console
    print("\n" + report)

    # Save detailed metrics as JSON
    json_file = args.output_dir / f"portfolio_metrics_max{args.max_positions}.json"
    with open(json_file, 'w') as f:
        # Convert non-serializable types
        json_metrics = {k: v for k, v in metrics.items()
                       if k not in ['monthly_returns', 'yearly_returns']}
        json_metrics['monthly_returns'] = metrics['monthly_returns']
        json_metrics['yearly_returns'] = metrics['yearly_returns']
        json.dump(json_metrics, f, indent=2, default=str)

    logger.info(f"\nDetailed metrics saved to: {json_file}")
    logger.info("Analysis complete!")


if __name__ == '__main__':
    main()
