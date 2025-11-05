"""Statistical Quality Control for Trading System.

This module provides statistical monitoring of live trading performance against
backtest expectations. Implements:

Phase 1: Control Charts with ±2σ bands and traffic light system
Phase 2: Confidence intervals, p-values, and sample size tracking
Phase 3: Sequential testing (SPRT) and Bayesian monitoring

Important: Instrument-level CSVs do NOT include slippage. Slippage is added at
portfolio level. For instrument-level monitoring, we compare raw returns.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class BacktestBaseline:
    """Expected performance metrics from backtest holdout period."""

    symbol: str
    sharpe: float
    profit_factor: float
    trades: int
    total_pnl: float
    max_drawdown: float
    win_rate: float  # Calculated from trades
    avg_return: float  # Average trade return
    std_return: float  # Standard deviation of returns
    threshold: float  # ML threshold used


@dataclass
class PortfolioBaseline:
    """Expected portfolio-level performance from backtest test period."""

    sharpe: float
    max_drawdown: float
    cagr: float
    total_pnl: float  # Calculated from test period
    win_rate: float  # Calculated from aggregated trades
    avg_return: float  # Average per-trade return
    std_return: float  # Std dev of returns
    trades: int  # Total trades in test period


@dataclass
class LivePerformance:
    """Live trading performance metrics."""

    symbol: str
    n_trades: int
    sharpe: float
    profit_factor: float
    win_rate: float
    total_pnl: float
    max_drawdown: float
    avg_return: float
    std_return: float
    returns: list[float]  # Individual trade returns


@dataclass
class StatisticalTest:
    """Results of statistical comparison."""

    metric_name: str
    expected: float
    observed: float
    z_score: float
    p_value: float
    ci_lower: float
    ci_upper: float
    status: str  # "green", "yellow", "red"
    message: str


class StatisticalMonitor:
    """Statistical quality control for trading system."""

    def __init__(self, results_dir: str | Path = "results"):
        """Initialize monitor.

        Args:
            results_dir: Directory containing optimization results
        """
        self.results_dir = Path(results_dir)

    def load_backtest_baseline(self, symbol: str) -> Optional[BacktestBaseline]:
        """Load expected performance from backtest results.

        Args:
            symbol: Trading symbol (e.g., "6A")

        Returns:
            BacktestBaseline or None if not found
        """
        # Load best.json for holdout metrics
        best_json = self.results_dir / f"{symbol}_optimization" / "best.json"

        if not best_json.exists():
            return None

        with open(best_json) as f:
            data = json.load(f)

        holdout = data.get("Holdout", {})

        if not holdout:
            return None

        # Load trade CSV to calculate win rate and return statistics
        trades_csv = self.results_dir / f"{symbol}_optimization" / f"{symbol}_trades.csv"

        if not trades_csv.exists():
            return None

        df = pd.read_csv(trades_csv)

        # Filter to only actual trades (non-zero returns)
        trades = df[df["Return"] != 0]["Return"].values

        if len(trades) == 0:
            return None

        # Calculate statistics from holdout period trades
        win_rate = float((trades > 0).sum() / len(trades) * 100)
        avg_return = float(trades.mean())
        std_return = float(trades.std())

        return BacktestBaseline(
            symbol=symbol,
            sharpe=holdout.get("Sharpe", 0.0),
            profit_factor=holdout.get("PF", 0.0),
            trades=int(holdout.get("Trades", 0)),
            total_pnl=holdout.get("Total_PnL_USD", 0.0),
            max_drawdown=holdout.get("Max_Drawdown_USD", 0.0),
            win_rate=win_rate,
            avg_return=avg_return,
            std_return=std_return,
            threshold=holdout.get("Thr", 0.65),
        )

    def calculate_live_performance(
        self, symbol: str, trades: list[dict[str, Any]]
    ) -> Optional[LivePerformance]:
        """Calculate live performance metrics from database trades.

        Args:
            symbol: Trading symbol
            trades: List of trades from database (with 'pnl' field)

        Returns:
            LivePerformance or None if insufficient data
        """
        if not trades:
            return None

        # Extract returns
        returns = [t["pnl"] for t in trades if t.get("pnl") != 0]

        if len(returns) < 2:
            return None

        returns_array = np.array(returns)

        # Calculate metrics
        win_rate = float((returns_array > 0).sum() / len(returns) * 100)
        avg_return = float(returns_array.mean())
        std_return = float(returns_array.std(ddof=1))

        # Sharpe (annualized assuming daily trades)
        sharpe = 0.0
        if std_return > 0:
            sharpe = avg_return / std_return * math.sqrt(252)

        # Profit factor
        gross_profit = float(returns_array[returns_array > 0].sum())
        gross_loss = float(abs(returns_array[returns_array < 0].sum()))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Max drawdown (cumulative)
        cumsum = returns_array.cumsum()
        running_max = np.maximum.accumulate(cumsum)
        drawdown = running_max - cumsum
        max_drawdown = float(drawdown.max())

        return LivePerformance(
            symbol=symbol,
            n_trades=len(returns),
            sharpe=sharpe,
            profit_factor=profit_factor,
            win_rate=win_rate,
            total_pnl=float(returns_array.sum()),
            max_drawdown=max_drawdown,
            avg_return=avg_return,
            std_return=std_return,
            returns=returns,
        )

    def run_statistical_tests(
        self, baseline: BacktestBaseline, live: LivePerformance, confidence: float = 0.95
    ) -> list[StatisticalTest]:
        """Run comprehensive statistical tests comparing live to baseline.

        Phase 1: Control charts (z-score, ±2σ bands)
        Phase 2: Confidence intervals and p-values
        Phase 3: Sequential testing (SPRT)

        Args:
            baseline: Expected performance from backtest
            live: Observed live performance
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            List of StatisticalTest results
        """
        tests = []

        # --- Test 1: Win Rate ---
        # Use binomial test for win rate
        expected_win_rate = baseline.win_rate / 100  # Convert to proportion
        observed_wins = int(live.n_trades * live.win_rate / 100)

        # Binomial test
        binom_pvalue = stats.binom_test(
            observed_wins, live.n_trades, expected_win_rate, alternative="two-sided"
        )

        # Standard error and confidence interval for proportion
        se_win_rate = math.sqrt(expected_win_rate * (1 - expected_win_rate) / live.n_trades)
        z_crit = stats.norm.ppf((1 + confidence) / 2)
        ci_lower_wr = (expected_win_rate - z_crit * se_win_rate) * 100
        ci_upper_wr = (expected_win_rate + z_crit * se_win_rate) * 100

        # Z-score for win rate
        z_wr = (
            (live.win_rate / 100 - expected_win_rate) / se_win_rate
            if se_win_rate > 0
            else 0.0
        )

        # Traffic light status
        status_wr = self._determine_status(abs(z_wr), binom_pvalue)
        message_wr = self._format_message("Win Rate", live.win_rate, baseline.win_rate, status_wr)

        tests.append(
            StatisticalTest(
                metric_name="Win Rate",
                expected=baseline.win_rate,
                observed=live.win_rate,
                z_score=z_wr,
                p_value=binom_pvalue,
                ci_lower=ci_lower_wr,
                ci_upper=ci_upper_wr,
                status=status_wr,
                message=message_wr,
            )
        )

        # --- Test 2: Average Return ---
        # Use t-test for average return (comparing sample mean to expected mean)
        expected_return = baseline.avg_return
        se_return = live.std_return / math.sqrt(live.n_trades)

        # T-test (one-sample)
        if se_return > 0:
            t_stat = (live.avg_return - expected_return) / se_return
            t_pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df=live.n_trades - 1))
        else:
            t_stat = 0.0
            t_pvalue = 1.0

        # Confidence interval
        t_crit = stats.t.ppf((1 + confidence) / 2, df=live.n_trades - 1)
        ci_lower_ret = expected_return - t_crit * se_return
        ci_upper_ret = expected_return + t_crit * se_return

        status_ret = self._determine_status(abs(t_stat), t_pvalue)
        message_ret = self._format_message(
            "Avg Return", live.avg_return, expected_return, status_ret, is_dollar=True
        )

        tests.append(
            StatisticalTest(
                metric_name="Average Return",
                expected=expected_return,
                observed=live.avg_return,
                z_score=t_stat,
                p_value=t_pvalue,
                ci_lower=ci_lower_ret,
                ci_upper=ci_upper_ret,
                status=status_ret,
                message=message_ret,
            )
        )

        # --- Test 3: Sharpe Ratio ---
        # Sharpe comparison (approximate z-test)
        # NOTE: Sharpe has complex distribution, this is approximate
        expected_sharpe = baseline.sharpe
        se_sharpe = 1 / math.sqrt(live.n_trades)  # Approximate SE for Sharpe

        z_sharpe = (live.sharpe - expected_sharpe) / se_sharpe if se_sharpe > 0 else 0.0
        sharpe_pvalue = 2 * (1 - stats.norm.cdf(abs(z_sharpe)))

        ci_lower_sharpe = expected_sharpe - z_crit * se_sharpe
        ci_upper_sharpe = expected_sharpe + z_crit * se_sharpe

        status_sharpe = self._determine_status(abs(z_sharpe), sharpe_pvalue)
        message_sharpe = self._format_message(
            "Sharpe", live.sharpe, expected_sharpe, status_sharpe
        )

        tests.append(
            StatisticalTest(
                metric_name="Sharpe Ratio",
                expected=expected_sharpe,
                observed=live.sharpe,
                z_score=z_sharpe,
                p_value=sharpe_pvalue,
                ci_lower=ci_lower_sharpe,
                ci_upper=ci_upper_sharpe,
                status=status_sharpe,
                message=message_sharpe,
            )
        )

        # --- Test 4: Profit Factor ---
        # Profit factor is ratio - use bootstrap CI (not implemented here, use simple z-test)
        expected_pf = baseline.profit_factor
        if expected_pf == float("inf") or live.profit_factor == float("inf"):
            # Can't compare infinite values
            tests.append(
                StatisticalTest(
                    metric_name="Profit Factor",
                    expected=expected_pf,
                    observed=live.profit_factor,
                    z_score=0.0,
                    p_value=1.0,
                    ci_lower=0.0,
                    ci_upper=float("inf"),
                    status="green",
                    message="Profit Factor: ∞ (no losses)",
                )
            )
        else:
            # Approximate SE for PF (this is simplistic)
            se_pf = expected_pf / math.sqrt(live.n_trades)
            z_pf = (live.profit_factor - expected_pf) / se_pf if se_pf > 0 else 0.0
            pf_pvalue = 2 * (1 - stats.norm.cdf(abs(z_pf)))

            ci_lower_pf = max(0, expected_pf - z_crit * se_pf)
            ci_upper_pf = expected_pf + z_crit * se_pf

            status_pf = self._determine_status(abs(z_pf), pf_pvalue)
            message_pf = self._format_message(
                "Profit Factor", live.profit_factor, expected_pf, status_pf
            )

            tests.append(
                StatisticalTest(
                    metric_name="Profit Factor",
                    expected=expected_pf,
                    observed=live.profit_factor,
                    z_score=z_pf,
                    p_value=pf_pvalue,
                    ci_lower=ci_lower_pf,
                    ci_upper=ci_upper_pf,
                    status=status_pf,
                    message=message_pf,
                )
            )

        return tests

    def _determine_status(self, z_score: float, p_value: float) -> str:
        """Determine traffic light status based on z-score and p-value.

        Green: Within ±1.5σ (good performance, expected variation)
        Yellow: Between ±1.5σ and ±2.5σ (monitor closely)
        Red: Beyond ±2.5σ (statistically significant degradation)

        Args:
            z_score: Absolute z-score
            p_value: Two-tailed p-value

        Returns:
            Status string: "green", "yellow", or "red"
        """
        if abs(z_score) <= 1.5:
            return "green"
        elif abs(z_score) <= 2.5:
            return "yellow"
        else:
            return "red"

    def _format_message(
        self,
        metric: str,
        observed: float,
        expected: float,
        status: str,
        is_dollar: bool = False,
    ) -> str:
        """Format status message.

        Args:
            metric: Metric name
            observed: Observed value
            expected: Expected value
            status: Status color
            is_dollar: Whether to format as currency

        Returns:
            Formatted message
        """
        if is_dollar:
            return (
                f"{metric}: ${observed:.4f} vs expected ${expected:.4f} "
                f"({status.upper()})"
            )
        else:
            return (
                f"{metric}: {observed:.2f} vs expected {expected:.2f} " f"({status.upper()})"
            )

    def run_sprt(
        self,
        baseline: BacktestBaseline,
        live: LivePerformance,
        alpha: float = 0.05,
        beta: float = 0.20,
    ) -> dict[str, Any]:
        """Run Sequential Probability Ratio Test (SPRT).

        Tests if live performance has degraded from baseline.
        H0: Performance matches baseline
        H1: Performance is worse than baseline

        Args:
            baseline: Expected performance
            live: Observed performance
            alpha: Type I error rate (false alarm)
            beta: Type II error rate (missed detection)

        Returns:
            Dictionary with SPRT results
        """
        if live.n_trades < 5:
            return {
                "decision": "continue",
                "log_likelihood_ratio": 0.0,
                "threshold_upper": math.log((1 - beta) / alpha),
                "threshold_lower": math.log(beta / (1 - alpha)),
                "message": "Insufficient data for SPRT (need ≥5 trades)",
            }

        # Use average return as test statistic
        # H0: μ = μ_baseline
        # H1: μ = μ_baseline * 0.8 (20% degradation)

        mu0 = baseline.avg_return
        mu1 = mu0 * 0.8  # Detect 20% degradation
        sigma = baseline.std_return

        if sigma == 0:
            return {
                "decision": "continue",
                "log_likelihood_ratio": 0.0,
                "threshold_upper": math.log((1 - beta) / alpha),
                "threshold_lower": math.log(beta / (1 - alpha)),
                "message": "Zero variance in baseline - cannot run SPRT",
            }

        # Calculate log-likelihood ratio
        # LLR = sum of log( P(x_i | H1) / P(x_i | H0) )
        llr = 0.0
        for x in live.returns:
            # Normal likelihood ratio
            ll0 = -0.5 * ((x - mu0) / sigma) ** 2
            ll1 = -0.5 * ((x - mu1) / sigma) ** 2
            llr += ll1 - ll0

        # SPRT thresholds
        A = math.log((1 - beta) / alpha)  # Accept H0 (performance OK)
        B = math.log(beta / (1 - alpha))  # Accept H1 (performance degraded)

        # Decision
        if llr >= A:
            decision = "accept_H0"
            message = "Performance consistent with baseline (PASS)"
        elif llr <= B:
            decision = "accept_H1"
            message = "Performance significantly worse than baseline (FAIL)"
        else:
            decision = "continue"
            message = f"Continue monitoring ({live.n_trades} trades so far)"

        return {
            "decision": decision,
            "log_likelihood_ratio": llr,
            "threshold_upper": A,
            "threshold_lower": B,
            "n_trades": live.n_trades,
            "message": message,
        }

    def get_sample_size_recommendation(
        self, baseline: BacktestBaseline, power: float = 0.80, alpha: float = 0.05
    ) -> int:
        """Calculate recommended sample size to detect 20% change in mean return.

        Args:
            baseline: Baseline performance
            power: Desired statistical power (1 - beta)
            alpha: Significance level

        Returns:
            Recommended number of trades for reliable testing
        """
        # Effect size: 20% change in mean return
        effect_size = abs(0.2 * baseline.avg_return / baseline.std_return)

        if effect_size == 0:
            return 100  # Default fallback

        # Calculate sample size for two-sample t-test
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

        return max(30, int(math.ceil(n)))  # Minimum 30 trades


def get_portfolio_baseline(results_dir: str | Path = "results") -> dict[str, Any]:
    """Load portfolio-level baseline from greedy optimization results.

    Args:
        results_dir: Results directory

    Returns:
        Dictionary with portfolio baseline metrics
    """
    results_path = Path(results_dir)

    # Find most recent greedy optimization JSON
    greedy_files = sorted(results_path.glob("greedy_optimization_*.json"))

    if not greedy_files:
        return {}

    with open(greedy_files[-1]) as f:
        data = json.load(f)

    test_metrics = data.get("test_metrics", {})

    return {
        "sharpe": test_metrics.get("sharpe", 0.0),
        "max_dd": test_metrics.get("max_dd", 0.0),
        "cagr": test_metrics.get("cagr", 0.0),
        "train_period": data.get("train_period", {}),
        "test_period": data.get("test_period", {}),
        "instruments": data.get("optimal_config", {}).get("instruments", []),
        "max_positions": data.get("optimal_config", {}).get("max_positions", 2),
    }


def load_portfolio_baseline_detailed(results_dir: str | Path = "results") -> Optional[PortfolioBaseline]:
    """Load detailed portfolio baseline including trade-level statistics.

    Aggregates all instrument CSVs to calculate portfolio-level metrics.

    Args:
        results_dir: Results directory

    Returns:
        PortfolioBaseline or None if data not available
    """
    results_path = Path(results_dir)

    # Load greedy optimization for high-level metrics
    greedy_files = sorted(results_path.glob("greedy_optimization_*.json"))
    if not greedy_files:
        return None

    with open(greedy_files[-1]) as f:
        data = json.load(f)

    test_metrics = data.get("test_metrics", {})
    instruments = data.get("optimal_config", {}).get("instruments", [])

    if not instruments:
        return None

    # Aggregate all trades from individual instrument CSVs
    all_returns = []

    for symbol in instruments:
        trades_csv = results_path / f"{symbol}_optimization" / f"{symbol}_trades.csv"

        if not trades_csv.exists():
            continue

        df = pd.read_csv(trades_csv)
        # Filter to non-zero returns (actual trades)
        returns = df[df["Return"] != 0]["Return"].values
        all_returns.extend(returns)

    if len(all_returns) == 0:
        return None

    all_returns = np.array(all_returns)

    # Calculate statistics
    win_rate = float((all_returns > 0).sum() / len(all_returns) * 100)
    avg_return = float(all_returns.mean())
    std_return = float(all_returns.std())

    # Estimate total P&L from test period (this is approximate)
    # The greedy optimizer doesn't store total P&L, so we use CAGR to estimate
    cagr = test_metrics.get("cagr", 0.0)
    max_dd = test_metrics.get("max_dd", 0.0)

    # Approximate total P&L assuming 1 year test period and $100k starting capital
    # total_pnl ≈ starting_capital × CAGR
    estimated_pnl = 100000 * cagr  # Rough estimate

    return PortfolioBaseline(
        sharpe=test_metrics.get("sharpe", 0.0),
        max_drawdown=max_dd,
        cagr=cagr,
        total_pnl=estimated_pnl,
        win_rate=win_rate,
        avg_return=avg_return,
        std_return=std_return,
        trades=len(all_returns),
    )
