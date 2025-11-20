"""
Vectorized backtester for Strategy Factory.

Fast, event-driven backtesting with proper position management,
slippage, commissions, and exit hierarchy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging
import sys
from pathlib import Path

# Add src to path for contract specs
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
try:
    from strategy.contract_specs import CONTRACT_SPECS
except ImportError:
    # Fallback if contract_specs not available
    CONTRACT_SPECS = {}

from ..strategies.base import BaseStrategy, TradeExit

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a completed trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    bars_held: int
    exit_type: str  # 'signal', 'stop_loss', 'take_profit', 'time', 'eod'
    direction: int = 1  # 1=long, -1=short (future)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BacktestResults:
    """
    Results from a backtest run.
    """
    # Strategy info
    strategy_name: str
    strategy_id: int
    symbol: str
    params: Dict[str, Any]

    # Date range
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    total_bars: int

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # P&L statistics
    total_pnl: float
    total_pnl_pct: float
    avg_pnl_per_trade: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

    # Risk metrics
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    profit_factor: float  # Gross profit / Gross loss

    # Holding statistics
    avg_bars_held: float
    max_bars_held: int
    min_bars_held: int

    # Exit breakdown
    exit_counts: Dict[str, int]  # Count by exit type

    # Trade list
    trades: List[Trade]

    # Equity curve
    equity_curve: pd.Series

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for database storage)."""
        # Helper to convert numpy types to native Python types
        def to_python(val):
            if hasattr(val, 'item'):  # numpy scalar
                return val.item()
            elif isinstance(val, (int, float, str, bool)):
                return val
            elif isinstance(val, pd.Timestamp):
                return str(val)
            else:
                return val

        result = {
            'strategy_name': str(self.strategy_name),
            'strategy_id': int(self.strategy_id),
            'symbol': str(self.symbol),
            'params': str(self.params),
            'start_date': str(self.start_date),
            'end_date': str(self.end_date),
            'total_bars': int(self.total_bars),
            'total_trades': int(self.total_trades),
            'winning_trades': int(self.winning_trades),
            'losing_trades': int(self.losing_trades),
            'win_rate': float(self.win_rate),
            'total_pnl': float(self.total_pnl),
            'total_pnl_pct': float(self.total_pnl_pct),
            'avg_pnl_per_trade': float(self.avg_pnl_per_trade),
            'avg_win': float(self.avg_win),
            'avg_loss': float(self.avg_loss),
            'largest_win': float(self.largest_win),
            'largest_loss': float(self.largest_loss),
            'sharpe_ratio': float(self.sharpe_ratio),
            'max_drawdown': float(self.max_drawdown),
            'max_drawdown_pct': float(self.max_drawdown_pct),
            'profit_factor': float(self.profit_factor),
            'avg_bars_held': float(self.avg_bars_held),
            'max_bars_held': int(self.max_bars_held),
            'min_bars_held': int(self.min_bars_held),
            'exit_counts': str(self.exit_counts)
        }
        return result

    def summary(self) -> str:
        """Generate text summary of results."""
        summary = f"""
{'=' * 80}
BACKTEST RESULTS: {self.strategy_name}
{'=' * 80}

Symbol: {self.symbol}
Parameters: {self.params}
Date Range: {self.start_date.date()} to {self.end_date.date()}
Total Bars: {self.total_bars:,}

TRADE STATISTICS
----------------
Total Trades: {self.total_trades:,}
Winning Trades: {self.winning_trades:,}
Losing Trades: {self.losing_trades:,}
Win Rate: {self.win_rate:.2%}

P&L STATISTICS
--------------
Total P&L: ${self.total_pnl:,.2f}
Total P&L %: {self.total_pnl_pct:.2%}
Avg P&L per Trade: ${self.avg_pnl_per_trade:.2f}
Avg Win: ${self.avg_win:.2f}
Avg Loss: ${self.avg_loss:.2f}
Largest Win: ${self.largest_win:.2f}
Largest Loss: ${self.largest_loss:.2f}

RISK METRICS
------------
Sharpe Ratio: {self.sharpe_ratio:.3f}
Max Drawdown: ${self.max_drawdown:,.2f} ({self.max_drawdown_pct:.2%})
Profit Factor: {self.profit_factor:.3f}

HOLDING STATISTICS
------------------
Avg Bars Held: {self.avg_bars_held:.1f}
Max Bars Held: {self.max_bars_held}
Min Bars Held: {self.min_bars_held}

EXIT BREAKDOWN
--------------
"""
        for exit_type, count in self.exit_counts.items():
            pct = count / self.total_trades * 100 if self.total_trades > 0 else 0
            summary += f"{exit_type}: {count:,} ({pct:.1f}%)\n"

        summary += "=" * 80

        return summary


class Backtester:
    """
    Event-driven backtester for research strategies.

    Features:
    - Proper position management (one position at a time)
    - Exit hierarchy (strategy → stop → target → time → EOD)
    - Contract-specific slippage and commission modeling
    - Equity curve tracking
    - Detailed trade analytics
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_per_side: float = 1.00,
        slippage_ticks: float = 1.0  # Number of ticks slippage per side
    ):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital
            commission_per_side: Commission per side (round-turn = 2x)
            slippage_ticks: Number of ticks slippage per side (default: 1 tick)
        """
        self.initial_capital = initial_capital
        self.commission_per_side = commission_per_side
        self.slippage_ticks = slippage_ticks

    def run(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        symbol: str
    ) -> BacktestResults:
        """
        Run backtest for a strategy.

        Args:
            strategy: Strategy instance with parameters set
            data: OHLCV dataframe
            symbol: Symbol being traded

        Returns:
            BacktestResults object with full analytics
        """
        # Get contract specs for tick-based slippage
        spec = CONTRACT_SPECS.get(symbol.upper(), {"tick_size": 0.01, "tick_value": 1.0})
        tick_size = spec["tick_size"]
        slippage_points = tick_size * self.slippage_ticks

        # Prepare data (calculate indicators)
        data = strategy.prepare_data(data.copy())

        # Skip warmup period
        warmup = strategy.warmup_period
        data = data.iloc[warmup:].reset_index(drop=False)

        # Generate entry signals
        entries = strategy.entry_logic(data, strategy.params)

        # Initialize tracking variables
        trades: List[Trade] = []
        in_position = False
        entry_idx = None
        entry_price = None
        entry_time = None

        capital = self.initial_capital
        equity_curve = []

        # Event loop
        for i in range(len(data)):
            current_bar = data.iloc[i]
            current_time = current_bar['datetime']
            current_price = current_bar['Close']

            # Check for exit if in position
            if in_position:
                trade_exit = strategy.get_exit(
                    data, entry_idx, entry_price, i, direction=1
                )

                if trade_exit.exit:
                    # Exit trade
                    exit_price = trade_exit.exit_price

                    # Apply slippage (1 tick worse on exit)
                    exit_price -= slippage_points

                    # Calculate P&L
                    pnl_points = exit_price - entry_price
                    pnl_pct = pnl_points / entry_price
                    commission = 2 * self.commission_per_side  # Round turn
                    pnl = pnl_points - commission

                    # Create trade record
                    trade = Trade(
                        entry_time=entry_time,
                        exit_time=current_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        bars_held=i - entry_idx,
                        exit_type=trade_exit.exit_type
                    )
                    trades.append(trade)

                    # Update capital
                    capital += pnl

                    # Reset position
                    in_position = False
                    entry_idx = None
                    entry_price = None
                    entry_time = None

            # Check for new entry if not in position
            elif entries.iloc[i] and not in_position:
                # Entry signal
                entry_price = current_price

                # Apply slippage (1 tick worse on entry)
                entry_price += slippage_points

                entry_time = current_time
                entry_idx = i
                in_position = True

            # Track equity
            equity_curve.append(capital)

        # Calculate results
        results = self._calculate_results(
            strategy, symbol, data, trades, equity_curve
        )

        return results

    def _calculate_results(
        self,
        strategy: BaseStrategy,
        symbol: str,
        data: pd.DataFrame,
        trades: List[Trade],
        equity_curve: List[float]
    ) -> BacktestResults:
        """
        Calculate backtest results from trades.

        Args:
            strategy: Strategy instance
            symbol: Symbol traded
            data: Price data
            trades: List of completed trades
            equity_curve: Equity curve over time

        Returns:
            BacktestResults object
        """
        if not trades:
            # No trades - return empty results
            return BacktestResults(
                strategy_name=strategy.name,
                strategy_id=strategy.strategy_id,
                symbol=symbol,
                params=strategy.params,
                start_date=data['datetime'].iloc[0],
                end_date=data['datetime'].iloc[-1],
                total_bars=len(data),
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                total_pnl_pct=0.0,
                avg_pnl_per_trade=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                max_drawdown_pct=0.0,
                profit_factor=0.0,
                avg_bars_held=0.0,
                max_bars_held=0,
                min_bars_held=0,
                exit_counts={},
                trades=[],
                equity_curve=pd.Series(equity_curve)
            )

        # Trade statistics
        total_trades = len(trades)
        pnls = [t.pnl for t in trades]
        winning_trades = sum(1 for pnl in pnls if pnl > 0)
        losing_trades = sum(1 for pnl in pnls if pnl < 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # P&L statistics
        total_pnl = sum(pnls)
        total_pnl_pct = total_pnl / self.initial_capital
        avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0

        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe ratio (annualized)
        returns = pd.Series(pnls)
        sharpe_ratio = self._calculate_sharpe(returns)

        # Drawdown
        equity_series = pd.Series(equity_curve)
        max_drawdown, max_drawdown_pct = self._calculate_drawdown(equity_series)

        # Holding statistics
        bars_held = [t.bars_held for t in trades]
        avg_bars_held = np.mean(bars_held)
        max_bars_held = max(bars_held)
        min_bars_held = min(bars_held)

        # Exit breakdown
        exit_counts = {}
        for trade in trades:
            exit_type = trade.exit_type
            exit_counts[exit_type] = exit_counts.get(exit_type, 0) + 1

        return BacktestResults(
            strategy_name=strategy.name,
            strategy_id=strategy.strategy_id,
            symbol=symbol,
            params=strategy.params,
            start_date=data['datetime'].iloc[0],
            end_date=data['datetime'].iloc[-1],
            total_bars=len(data),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            avg_pnl_per_trade=avg_pnl_per_trade,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            profit_factor=profit_factor,
            avg_bars_held=avg_bars_held,
            max_bars_held=max_bars_held,
            min_bars_held=min_bars_held,
            exit_counts=exit_counts,
            trades=trades,
            equity_curve=equity_series
        )

    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio (annualized).

        Assumes daily returns (adjust for intraday if needed).

        Args:
            returns: Series of trade returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        mean_return = returns.mean()
        std_return = returns.std()

        if std_return == 0:
            return 0.0

        # Sharpe = (mean - rf) / std
        # Annualize: multiply by sqrt(252) for daily, sqrt(12) for monthly
        # For trades, we don't annualize since frequency varies
        sharpe = (mean_return - risk_free_rate) / std_return

        return sharpe

    def _calculate_drawdown(self, equity: pd.Series) -> tuple[float, float]:
        """
        Calculate maximum drawdown.

        Args:
            equity: Equity curve

        Returns:
            Tuple of (max_drawdown_dollars, max_drawdown_pct)
        """
        if len(equity) == 0:
            return 0.0, 0.0

        # Calculate running maximum
        running_max = equity.expanding().max()

        # Drawdown = current - running_max
        drawdown = equity - running_max

        # Max drawdown
        max_dd = drawdown.min()
        max_dd_pct = max_dd / running_max.max() if running_max.max() > 0 else 0

        return abs(max_dd), abs(max_dd_pct)


if __name__ == "__main__":
    """
    Test backtester with RSI(2) strategy.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from strategy_factory.engine.data_loader import load_data
    from strategy_factory.strategies.rsi2_mean_reversion import RSI2MeanReversion
    import logging

    logging.basicConfig(level=logging.INFO)

    print("Testing Backtester with RSI(2) Strategy")
    print("=" * 80)
    print()

    # Load ES data (2023 for quick test)
    data = load_data("ES", "15min", "2023-01-01", "2023-12-31")
    print(f"Loaded {len(data):,} bars")
    print()

    # Create strategy
    strategy = RSI2MeanReversion(params={
        'rsi_length': 2,
        'rsi_oversold': 10,
        'rsi_overbought': 65,
        'stop_loss_atr': 1.0,
        'take_profit_atr': 1.0
    })

    # Run backtest
    backtester = Backtester(
        initial_capital=100000,
        commission_per_side=2.50,
        slippage_pct=0.0001
    )

    results = backtester.run(strategy, data, "ES")

    # Print summary
    print(results.summary())
