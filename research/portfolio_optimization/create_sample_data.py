#!/usr/bin/env python3
"""
Generate Sample Validation Data for Portfolio Optimizer Testing

Creates synthetic equity curves for multiple strategies to test the portfolio
optimization system without requiring real ML pipeline results.

Generates realistic equity curves with:
- Different Sharpe ratios
- Various drawdown profiles
- Correlation between strategies
- Daily P&L variations

Author: Rooney Capital
Date: 2025-01-22
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)


def generate_equity_curve(
    start_date: str,
    end_date: str,
    sharpe: float,
    volatility: float,
    max_drawdown: float,
    drift: float = 0.0
) -> pd.DataFrame:
    """
    Generate synthetic equity curve with specified characteristics.

    Args:
        start_date: Start date
        end_date: End date
        sharpe: Target Sharpe ratio
        volatility: Daily volatility (std of daily returns)
        max_drawdown: Maximum drawdown in dollars
        drift: Daily drift (expected return)

    Returns:
        DataFrame with date, equity, daily_pnl columns
    """
    dates = pd.date_range(start_date, end_date, freq='D')

    # Generate returns with target Sharpe
    # Sharpe = E[R] / std(R) * sqrt(252)
    # So E[R] = Sharpe * std(R) / sqrt(252)
    daily_return = sharpe * volatility / np.sqrt(252)

    # Generate random daily P&L
    daily_pnl = np.random.normal(daily_return, volatility, len(dates))

    # Add some autocorrelation (realistic for trading strategies)
    for i in range(1, len(daily_pnl)):
        daily_pnl[i] += 0.2 * daily_pnl[i-1]

    # Calculate equity curve
    equity = np.cumsum(daily_pnl)

    # Apply drawdown constraint
    running_max = np.maximum.accumulate(equity)
    drawdown = running_max - equity

    # Scale if drawdown exceeds max
    if drawdown.max() > max_drawdown:
        scale_factor = max_drawdown / drawdown.max()
        daily_pnl *= scale_factor
        equity = np.cumsum(daily_pnl)

    df = pd.DataFrame({
        'date': dates,
        'equity': equity,
        'daily_pnl': daily_pnl
    })

    return df


def create_sample_strategies():
    """Create sample validation data for 10 strategies."""

    strategies = [
        # Strategy 1: ES_21 (RSI2) - High Sharpe, low DD
        {
            'id': 'ES_21',
            'symbol': 'ES',
            'strategy_num': 21,
            'name': 'RSI2 Mean Reversion',
            'sharpe_2022_2023': 2.8,
            'sharpe_2024': 2.5,
            'volatility': 150,
            'max_dd_2022_2023': 2100,
            'max_dd_2024': 1800
        },
        # Strategy 2: NQ_21 (RSI2) - Very high Sharpe, higher DD
        {
            'id': 'NQ_21',
            'symbol': 'NQ',
            'strategy_num': 21,
            'name': 'RSI2 Mean Reversion',
            'sharpe_2022_2023': 3.2,
            'sharpe_2024': 2.8,
            'volatility': 180,
            'max_dd_2022_2023': 2800,
            'max_dd_2024': 2400
        },
        # Strategy 3: GC_42 (Gap Down) - Medium Sharpe, low DD
        {
            'id': 'GC_42',
            'symbol': 'GC',
            'strategy_num': 42,
            'name': 'Gap Down Reversal',
            'sharpe_2022_2023': 2.1,
            'sharpe_2024': 1.9,
            'volatility': 120,
            'max_dd_2022_2023': 1500,
            'max_dd_2024': 1300
        },
        # Strategy 4: CL_45 (IBS) - Good diversifier
        {
            'id': 'CL_45',
            'symbol': 'CL',
            'strategy_num': 45,
            'name': 'IBS Strategy',
            'sharpe_2022_2023': 2.4,
            'sharpe_2024': 2.1,
            'volatility': 140,
            'max_dd_2022_2023': 1900,
            'max_dd_2024': 1600
        },
        # Strategy 5: 6E_37 (Double7s) - Lower Sharpe but good diversifier
        {
            'id': '6E_37',
            'symbol': '6E',
            'strategy_num': 37,
            'name': 'Double 7s',
            'sharpe_2022_2023': 1.8,
            'sharpe_2024': 1.6,
            'volatility': 100,
            'max_dd_2022_2023': 1200,
            'max_dd_2024': 1000
        },
        # Strategy 6: ES_37 (Double7s) - High correlation with ES_21
        {
            'id': 'ES_37',
            'symbol': 'ES',
            'strategy_num': 37,
            'name': 'Double 7s',
            'sharpe_2022_2023': 2.3,
            'sharpe_2024': 2.0,
            'volatility': 140,
            'max_dd_2022_2023': 2000,
            'max_dd_2024': 1700
        },
        # Strategy 7: YM_21 (RSI2) - Good but violates DD constraint when combined
        {
            'id': 'YM_21',
            'symbol': 'YM',
            'strategy_num': 21,
            'name': 'RSI2 Mean Reversion',
            'sharpe_2022_2023': 2.6,
            'sharpe_2024': 2.3,
            'volatility': 160,
            'max_dd_2022_2023': 2400,
            'max_dd_2024': 2000
        },
        # Strategy 8: SI_42 (Gap Down) - Low correlation
        {
            'id': 'SI_42',
            'symbol': 'SI',
            'strategy_num': 42,
            'name': 'Gap Down Reversal',
            'sharpe_2022_2023': 1.9,
            'sharpe_2024': 1.7,
            'volatility': 110,
            'max_dd_2022_2023': 1400,
            'max_dd_2024': 1200
        },
        # Strategy 9: RTY_45 (IBS) - Volatile but high return
        {
            'id': 'RTY_45',
            'symbol': 'RTY',
            'strategy_num': 45,
            'name': 'IBS Strategy',
            'sharpe_2022_2023': 2.7,
            'sharpe_2024': 2.4,
            'volatility': 170,
            'max_dd_2022_2023': 2600,
            'max_dd_2024': 2200
        },
        # Strategy 10: NG_40 (Buy5BarLow) - Would violate daily loss limit
        {
            'id': 'NG_40',
            'symbol': 'NG',
            'strategy_num': 40,
            'name': 'Buy on 5 Bar Low',
            'sharpe_2022_2023': 2.0,
            'sharpe_2024': 1.8,
            'volatility': 200,  # Very volatile - can have $1500 daily losses
            'max_dd_2022_2023': 3200,
            'max_dd_2024': 2800
        },
    ]

    print("Generating sample validation data for 10 strategies...")
    print("=" * 80)

    output_base = Path('test_data/ml_meta_labeling/results')

    for strat in strategies:
        print(f"\n{strat['id']} ({strat['name']})")

        # Create output directory
        strat_dir = output_base / strat['id']
        strat_dir.mkdir(parents=True, exist_ok=True)

        # Generate 2022-2023 equity curve
        df_2022_2023 = generate_equity_curve(
            start_date='2022-01-01',
            end_date='2023-12-31',
            sharpe=strat['sharpe_2022_2023'],
            volatility=strat['volatility'],
            max_drawdown=strat['max_dd_2022_2023']
        )

        # Generate 2024 equity curve
        df_2024 = generate_equity_curve(
            start_date='2024-01-01',
            end_date='2024-12-31',
            sharpe=strat['sharpe_2024'],
            volatility=strat['volatility'],
            max_drawdown=strat['max_dd_2024']
        )

        # Combine
        df_full = pd.concat([df_2022_2023, df_2024], ignore_index=True)

        # Save equity curve
        equity_file = strat_dir / f"{strat['id']}_validation_equity.csv"
        df_full.to_csv(equity_file, index=False)

        # Calculate metrics
        sharpe_full = (df_full['daily_pnl'].mean() / df_full['daily_pnl'].std()) * np.sqrt(252)
        running_max = np.maximum.accumulate(df_full['equity'])
        drawdown = running_max - df_full['equity']
        max_dd = drawdown.max()
        max_daily_loss = abs(df_full['daily_pnl'].min())

        print(f"  2022-2024 Metrics:")
        print(f"    Sharpe: {sharpe_full:.2f}")
        print(f"    Max DD: ${max_dd:,.0f}")
        print(f"    Max Daily Loss: ${max_daily_loss:,.0f}")
        print(f"    Total Return: ${df_full['equity'].iloc[-1]:,.0f}")
        print(f"  Saved to: {equity_file}")

    print("\n" + "=" * 80)
    print(f"✅ Generated validation data for {len(strategies)} strategies")
    print(f"Output directory: {output_base}")
    print("\nExpected optimization results:")
    print("  - Should select NQ_21 first (highest Sharpe)")
    print("  - Should add ES_21, GC_42, CL_45 for diversification")
    print("  - Should NOT add all 10 (would violate DD constraint)")
    print("  - Should avoid NG_40 (too volatile for daily loss limit)")


if __name__ == '__main__':
    create_sample_strategies()
