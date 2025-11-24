"""
Strategy #21: RSI(2) Mean Reversion

Classic Larry Connors strategy:
- Enter when RSI(2) < oversold threshold
- Exit when RSI(2) > overbought threshold

High-frequency mean reversion strategy with strong edge in equity indices.

Expected Performance (ES 2010-2024):
- Trade Count: 15,000+
- Raw Sharpe: 0.3-0.5
- ML Sharpe: 1.0-2.0+ (after meta-labeling)
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np
from .base import BaseStrategy, TradeExit, calculate_rsi


class RSI2MeanReversion(BaseStrategy):
    """
    RSI(2) Mean Reversion Strategy (Catalogue #21).

    Entry:
    - RSI(period) < oversold_threshold

    Exit:
    - RSI(period) > overbought_threshold
    - OR standard exits (stop/target/time/EOD)

    Parameters to optimize:
    - rsi_length: [2, 3, 4]
    - rsi_oversold: [5, 10, 15]
    - rsi_overbought: [60, 65, 70, 75]
    """

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(
            strategy_id=21,
            name="RSI2_MeanReversion",
            archetype="mean_reversion",
            params=params
        )

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """Parameter grid for optimization."""
        return {
            'rsi_length': [2, 3, 4],
            'rsi_oversold': [5, 10, 15],
            'rsi_overbought': [60, 65, 70, 75]
            # Note: stop_loss_atr and take_profit_atr fixed at 1.0 for Phase 1
        }

    @property
    def warmup_period(self) -> int:
        """Minimum bars needed before strategy can trade."""
        # Need enough bars to calculate RSI
        max_rsi_period = max(self.param_grid['rsi_length'])
        return max_rsi_period + 5  # +5 for buffer

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI indicator.

        Args:
            data: OHLCV dataframe

        Returns:
            DataFrame with 'rsi' column added
        """
        rsi_length = self.params.get('rsi_length', 2)
        data['rsi'] = calculate_rsi(data['Close'], period=rsi_length)

        return data

    def entry_logic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """
        Generate entry signals when RSI crosses below oversold threshold.

        Args:
            data: OHLCV dataframe with 'rsi' column
            params: Strategy parameters

        Returns:
            Boolean Series: True where entry signal occurs
        """
        rsi_oversold = params.get('rsi_oversold', 10)

        # Entry: RSI crosses below oversold threshold
        entry = data['rsi'] < rsi_oversold

        return entry

    def exit_logic(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any],
        entry_idx: int,
        entry_price: float,
        current_idx: int
    ) -> TradeExit:
        """
        Exit when RSI crosses above overbought threshold.

        Args:
            data: OHLCV dataframe with 'rsi' column
            params: Strategy parameters
            entry_idx: Entry bar index
            entry_price: Entry price
            current_idx: Current bar index

        Returns:
            TradeExit object
        """
        rsi_overbought = params.get('rsi_overbought', 65)

        current_bar = data.iloc[current_idx]
        current_rsi = current_bar['rsi']

        # Exit when RSI crosses above overbought
        if current_rsi > rsi_overbought:
            return TradeExit(
                exit=True,
                exit_type='signal',
                exit_price=current_bar['Close']
            )

        # No strategy-specific exit
        return TradeExit(exit=False, exit_type='none')


if __name__ == "__main__":
    """
    Test RSI(2) strategy on ES data.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from strategy_factory.engine.data_loader import load_data
    import logging

    logging.basicConfig(level=logging.INFO)

    print("Testing RSI(2) Mean Reversion Strategy")
    print("=" * 50)
    print()

    # Load ES data (2023 only for quick test)
    data = load_data("ES", "15min", "2023-01-01", "2023-12-31")
    print(f"Loaded {len(data):,} bars")
    print()

    # Create strategy instance
    strategy = RSI2MeanReversion(params={
        'rsi_length': 2,
        'rsi_oversold': 10,
        'rsi_overbought': 65
    })

    # Prepare data (calculate indicators)
    data_with_indicators = strategy.prepare_data(data)
    print("Indicators calculated:")
    print(f"  - RSI(2): min={data_with_indicators['rsi'].min():.1f}, "
          f"max={data_with_indicators['rsi'].max():.1f}")
    print(f"  - ATR(14): mean={data_with_indicators['atr'].mean():.2f}")
    print()

    # Generate entry signals
    entries = strategy.entry_logic(data_with_indicators, strategy.params)
    entry_count = entries.sum()
    print(f"Entry signals: {entry_count}")
    print(f"Entry rate: {entry_count / len(data) * 100:.2f}%")
    print()

    # Show first few entry signals
    entry_dates = data_with_indicators[entries].index[:5]
    print("First 5 entry signals:")
    for date in entry_dates:
        bar = data_with_indicators.loc[date]
        print(f"  {date}: Close={bar['Close']:.2f}, RSI={bar['rsi']:.1f}")
