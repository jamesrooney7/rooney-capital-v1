"""
Portfolio Constructor - Combines multiple symbol models into a unified portfolio.

This module provides functionality to:
1. Load optimization results from multiple symbols
2. Generate combined portfolio signals with position constraints
3. Optimize max_positions parameter to maximize Sharpe ratio
"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


@dataclass
class ModelResult:
    """Container for a single symbol's model and metadata."""
    symbol: str
    model: RandomForestClassifier
    features: List[str]
    threshold: float
    sharpe: float
    profit_factor: float
    trades: int
    metadata: dict


class PortfolioConstructor:
    """
    Combines multiple symbol models into a portfolio with position constraints.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing optimization results (e.g., 'results/' or 'src/models/')
    max_positions : int, optional
        Maximum number of positions to hold simultaneously. If None, no limit.
    ranking_method : str, default='probability'
        Method to rank signals when limiting positions:
        - 'probability': Rank by prediction probability
        - 'sharpe': Rank by symbol's historical Sharpe ratio
        - 'profit_factor': Rank by symbol's profit factor
    """

    def __init__(
        self,
        results_dir: str | Path,
        max_positions: Optional[int] = None,
        ranking_method: str = 'probability'
    ):
        self.results_dir = Path(results_dir)
        self.max_positions = max_positions
        self.ranking_method = ranking_method
        self.models: Dict[str, ModelResult] = {}

        if ranking_method not in ['probability', 'sharpe', 'profit_factor']:
            raise ValueError(f"Invalid ranking_method: {ranking_method}")

    def load_models(self, symbols: Optional[List[str]] = None) -> None:
        """
        Load model results from the results directory.

        Parameters
        ----------
        symbols : list of str, optional
            List of symbols to load. If None, loads all available symbols.
        """
        if symbols is None:
            symbols = self._discover_available_symbols()

        for symbol in symbols:
            try:
                model_result = self._load_symbol_model(symbol)
                self.models[symbol] = model_result
                print(f"Loaded {symbol}: Sharpe={model_result.sharpe:.3f}, "
                      f"Trades={model_result.trades}")
            except Exception as e:
                print(f"Warning: Failed to load {symbol}: {e}")

        print(f"\nSuccessfully loaded {len(self.models)} models")

    def _discover_available_symbols(self) -> List[str]:
        """Discover available symbols from the results directory."""
        symbols = []

        # Check for {SYMBOL}_optimization directories
        for item in self.results_dir.iterdir():
            if item.is_dir() and item.name.endswith('_optimization'):
                symbol = item.name.replace('_optimization', '')
                symbols.append(symbol)

        # Also check for {SYMBOL}_best.json files (src/models format)
        for item in self.results_dir.glob('*_best.json'):
            symbol = item.stem.replace('_best', '')
            if symbol not in symbols:
                symbols.append(symbol)

        return sorted(symbols)

    def _load_symbol_model(self, symbol: str) -> ModelResult:
        """Load a single symbol's model and metadata."""
        # Try loading from {SYMBOL}_optimization/ directory first
        opt_dir = self.results_dir / f"{symbol}_optimization"
        if opt_dir.exists():
            metadata_path = opt_dir / "best.json"
            model_path = opt_dir / f"{symbol}_rf_model.pkl"
        else:
            # Try src/models format
            metadata_path = self.results_dir / f"{symbol}_best.json"
            model_path = self.results_dir / f"{symbol}_rf_model.pkl"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found for {symbol}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found for {symbol}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load model
        with open(model_path, 'rb') as f:
            model_bundle = pickle.load(f)

        # Extract model and features (handle both formats)
        if isinstance(model_bundle, dict):
            model = model_bundle.get('model')
            features = model_bundle.get('features', metadata.get('Features', []))
        else:
            model = model_bundle
            features = metadata.get('Features', [])

        return ModelResult(
            symbol=symbol,
            model=model,
            features=features,
            threshold=metadata.get('Prod_Threshold', 0.5),
            sharpe=metadata.get('Sharpe', 0.0),
            profit_factor=metadata.get('Profit_Factor', 1.0),
            trades=metadata.get('Trades', 0),
            metadata=metadata
        )

    def generate_signals(
        self,
        feature_data: Dict[str, pd.DataFrame],
        returns_data: Dict[str, pd.Series]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate portfolio signals for all symbols.

        Parameters
        ----------
        feature_data : dict of {symbol: DataFrame}
            Feature data for each symbol. Index must be datetime.
        returns_data : dict of {symbol: Series}
            Forward returns for each symbol. Index must be datetime.

        Returns
        -------
        signals : DataFrame
            Boolean signals for each symbol (columns), datetime index
        probabilities : DataFrame
            Prediction probabilities for each symbol (columns), datetime index
        """
        all_signals = {}
        all_probabilities = {}

        for symbol, model_result in self.models.items():
            if symbol not in feature_data:
                print(f"Warning: No feature data for {symbol}")
                continue

            df = feature_data[symbol].copy()

            # Ensure all required features are present
            missing_features = set(model_result.features) - set(df.columns)
            if missing_features:
                print(f"Warning: {symbol} missing features: {missing_features}")
                continue

            # Generate predictions
            X = df[model_result.features]
            probas = model_result.model.predict_proba(X)[:, 1]
            signals = probas >= model_result.threshold

            all_signals[symbol] = pd.Series(signals, index=df.index)
            all_probabilities[symbol] = pd.Series(probas, index=df.index)

        signals_df = pd.DataFrame(all_signals)
        probas_df = pd.DataFrame(all_probabilities)

        # Apply max_positions constraint if specified
        if self.max_positions is not None:
            signals_df = self._apply_position_limit(
                signals_df, probas_df, returns_data
            )

        return signals_df, probas_df

    def _apply_position_limit(
        self,
        signals: pd.DataFrame,
        probabilities: pd.DataFrame,
        returns_data: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Apply max_positions constraint by ranking signals.

        When more than max_positions signals are active, keep only the top-ranked ones.
        """
        limited_signals = signals.copy()

        for idx in signals.index:
            active_signals = signals.loc[idx]
            n_active = active_signals.sum()

            if n_active > self.max_positions:
                # Rank signals based on chosen method
                if self.ranking_method == 'probability':
                    scores = probabilities.loc[idx]
                elif self.ranking_method == 'sharpe':
                    scores = pd.Series({
                        sym: self.models[sym].sharpe
                        for sym in active_signals[active_signals].index
                    })
                elif self.ranking_method == 'profit_factor':
                    scores = pd.Series({
                        sym: self.models[sym].profit_factor
                        for sym in active_signals[active_signals].index
                    })

                # Keep only top max_positions
                top_symbols = scores.nlargest(self.max_positions).index
                limited_signals.loc[idx] = False
                limited_signals.loc[idx, top_symbols] = True

        return limited_signals

    def backtest_portfolio(
        self,
        signals: pd.DataFrame,
        returns_data: Dict[str, pd.Series],
        initial_capital: float = 250000.0,
        position_size_pct: float = 0.95,
        commission_per_side: float = 1.25,
        slippage_pct: float = 0.0001
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """
        Backtest the portfolio strategy.

        Parameters
        ----------
        signals : DataFrame
            Boolean signals for each symbol (columns), datetime index
        returns_data : dict of {symbol: Series}
            Forward returns for each symbol
        initial_capital : float
            Starting capital
        position_size_pct : float
            Percentage of capital to allocate per position
        commission_per_side : float
            Commission per trade side in dollars
        slippage_pct : float
            Slippage as percentage of price

        Returns
        -------
        equity_curve : Series
            Portfolio equity over time
        metrics : dict
            Performance metrics (Sharpe, returns, etc.)
        """
        # Align all data to common index
        common_index = signals.index

        # Calculate position sizes (equal weight across active positions)
        portfolio_returns = []
        portfolio_dates = []
        n_positions_series = []

        equity = initial_capital

        for idx in common_index:
            active_symbols = signals.loc[idx]
            active_symbols = active_symbols[active_symbols].index.tolist()
            n_positions = len(active_symbols)
            n_positions_series.append(n_positions)

            if n_positions == 0:
                portfolio_returns.append(0.0)
                portfolio_dates.append(idx)
                continue

            # Calculate per-position allocation
            position_value = equity * position_size_pct / n_positions

            # Calculate portfolio return for this period
            period_return = 0.0

            for symbol in active_symbols:
                if symbol not in returns_data:
                    continue

                if idx not in returns_data[symbol].index:
                    continue

                symbol_return = returns_data[symbol].loc[idx]

                # Apply costs
                commission_pct = (commission_per_side * 2) / position_value
                total_cost = commission_pct + slippage_pct

                position_return = (symbol_return - total_cost) / n_positions
                period_return += position_return

            portfolio_returns.append(period_return)
            portfolio_dates.append(idx)
            equity *= (1 + period_return)

        # Convert to series
        returns_series = pd.Series(portfolio_returns, index=portfolio_dates)
        equity_curve = initial_capital * (1 + returns_series).cumprod()

        # Calculate metrics
        metrics = self._calculate_metrics(
            returns_series,
            equity_curve,
            n_positions_series
        )

        return equity_curve, metrics

    def _calculate_metrics(
        self,
        returns: pd.Series,
        equity: pd.Series,
        n_positions: List[int]
    ) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]

        # Annualized metrics (assuming daily data)
        periods_per_year = 252
        n_periods = len(returns)
        years = n_periods / periods_per_year

        annualized_return = (1 + total_return) ** (1 / years) - 1
        annualized_vol = returns.std() * np.sqrt(periods_per_year)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0

        # Additional metrics
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0

        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = abs(negative_returns.mean()) if len(negative_returns) > 0 else 0
        profit_factor = (positive_returns.sum() / abs(negative_returns.sum())
                        if negative_returns.sum() != 0 else np.inf)

        # Max drawdown
        cummax = equity.expanding().max()
        drawdown = (equity - cummax) / cummax
        max_drawdown = drawdown.min()

        # Average positions
        avg_positions = np.mean(n_positions)

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'n_periods': n_periods,
            'avg_positions': avg_positions
        }

    def optimize_max_positions(
        self,
        feature_data: Dict[str, pd.DataFrame],
        returns_data: Dict[str, pd.Series],
        position_range: Optional[Tuple[int, int]] = None,
        **backtest_kwargs
    ) -> pd.DataFrame:
        """
        Optimize max_positions parameter to maximize Sharpe ratio.

        Parameters
        ----------
        feature_data : dict of {symbol: DataFrame}
            Feature data for each symbol
        returns_data : dict of {symbol: Series}
            Forward returns for each symbol
        position_range : tuple of (min, max), optional
            Range of max_positions to test. Default is (1, n_symbols)
        **backtest_kwargs
            Additional arguments passed to backtest_portfolio

        Returns
        -------
        results : DataFrame
            Performance metrics for each max_positions value
        """
        n_symbols = len(self.models)

        if position_range is None:
            position_range = (1, n_symbols)

        min_pos, max_pos = position_range
        max_pos = min(max_pos, n_symbols)  # Can't have more positions than symbols

        results = []

        print(f"Testing max_positions from {min_pos} to {max_pos}...")
        print("-" * 80)

        for max_pos_value in range(min_pos, max_pos + 1):
            print(f"\nTesting max_positions = {max_pos_value}")

            # Update max_positions
            original_max = self.max_positions
            self.max_positions = max_pos_value

            # Generate signals with this constraint
            signals, probas = self.generate_signals(feature_data, returns_data)

            # Backtest
            equity, metrics = self.backtest_portfolio(
                signals, returns_data, **backtest_kwargs
            )

            # Store results
            result = {'max_positions': max_pos_value, **metrics}
            results.append(result)

            print(f"  Sharpe: {metrics['sharpe_ratio']:.3f} | "
                  f"Annual Return: {metrics['annualized_return']*100:.2f}% | "
                  f"Max DD: {metrics['max_drawdown']*100:.2f}%")

            # Restore original setting
            self.max_positions = original_max

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)

        print("\n" + "=" * 80)
        print("OPTIMIZATION RESULTS (sorted by Sharpe Ratio)")
        print("=" * 80)
        print(results_df.to_string(index=False))
        print("=" * 80)

        return results_df

    def get_summary(self) -> pd.DataFrame:
        """Get summary statistics of loaded models."""
        summary_data = []

        for symbol, model_result in self.models.items():
            summary_data.append({
                'Symbol': symbol,
                'Sharpe': model_result.sharpe,
                'Profit_Factor': model_result.profit_factor,
                'Trades': model_result.trades,
                'Threshold': model_result.threshold,
                'N_Features': len(model_result.features)
            })

        df = pd.DataFrame(summary_data)
        df = df.sort_values('Sharpe', ascending=False)

        return df
