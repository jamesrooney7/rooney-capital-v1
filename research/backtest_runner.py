#!/usr/bin/env python3
"""
Backtest runner using production IbsStrategy code.

This script ensures parity between backtesting and live trading by:
1. Using the EXACT production strategy class (src/strategy/ibs_strategy.py)
2. Loading resampled historical data (hourly + daily)
3. Applying same commission, slippage, and position sizing

Usage:
    # Backtest ES for 2023-2024
    python research/backtest_runner.py --symbol ES --start 2023-01-01 --end 2024-12-31

    # Backtest multiple symbols
    python research/backtest_runner.py --symbols ES NQ YM RTY --start 2023-01-01

    # Backtest with specific model
    python research/backtest_runner.py --symbol ES --model src/models/ES_v2.joblib
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging
import backtrader as bt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.utils.data_loader import setup_cerebro_with_data
from src.strategy.ibs_strategy import IbsStrategy
from src.models.loader import load_model_bundle
from src.config import COMMISSION_PER_SIDE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PercentageSizer(bt.Sizer):
    """
    Position sizer based on percentage of portfolio.

    Params:
        - percent: Percentage of portfolio to risk per trade (default: 100 = full position)
    """
    params = (('percent', 100),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        """Calculate position size based on available cash."""
        position_value = (cash * self.params.percent) / 100.0

        # Get price
        price = data.close[0]
        if price <= 0:
            return 0

        # Calculate number of contracts
        size = int(position_value / price)

        return max(size, 0)


def run_backtest(
    symbol: str,
    start_date: str,
    end_date: str,
    data_dir: str = 'data/resampled',
    initial_cash: float = 100000.0,
    commission: float = COMMISSION_PER_SIDE,
    strategy_params: dict = None,
    use_ml: bool = True,
):
    """
    Run backtest for a single symbol using production IbsStrategy.

    Automatically loads the trained ML model and filter configuration for the symbol
    from src/models/{SYMBOL}_rf_model.pkl and {SYMBOL}_best.json.

    Args:
        symbol: Symbol to backtest (e.g., 'ES', 'NQ')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        data_dir: Directory containing resampled data
        initial_cash: Starting capital
        commission: Commission per trade side
        strategy_params: Optional strategy parameters to override
        use_ml: Whether to load ML model (default: True)

    Returns:
        Backtest results
    """
    logger.info(f"Starting backtest for {symbol}")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Initial cash: ${initial_cash:,.2f}")
    logger.info(f"Commission: ${commission} per side")

    # Create Cerebro instance
    cerebro = bt.Cerebro()

    # Set initial cash
    cerebro.broker.setcash(initial_cash)

    # Set commission
    cerebro.broker.setcommission(commission=commission)

    # Load data for primary symbol and reference symbols
    # IbsStrategy needs TLT_day for regime filters
    symbols_to_load = [symbol, 'TLT']

    # Also load pair symbols if needed (e.g., NQ for ES)
    # For now, just load the main symbol + TLT
    # TODO: Add pair symbol logic based on PAIR_MAP

    logger.info(f"Loading data for symbols: {symbols_to_load}")
    setup_cerebro_with_data(
        cerebro,
        symbols=symbols_to_load,
        data_dir=data_dir,
        start_date=start_date,
        end_date=end_date
    )

    # Load ML model bundle for this symbol
    if use_ml:
        try:
            logger.info(f"Loading ML model bundle for {symbol}...")
            bundle = load_model_bundle(symbol)
            logger.info(f"✅ Loaded model with {len(bundle.features)} features, threshold={bundle.threshold}")
            logger.info(f"   Features: {', '.join(bundle.features[:5])}... ({len(bundle.features)} total)")

            # Get strategy parameters from bundle
            strat_params = bundle.strategy_kwargs()
            strat_params['symbol'] = symbol

        except FileNotFoundError as e:
            logger.warning(f"⚠️  No ML model found for {symbol}: {e}")
            logger.warning(f"   Running with default parameters (no ML filter)")
            strat_params = {'symbol': symbol}
    else:
        logger.info(f"ML disabled - running with default parameters")
        strat_params = {'symbol': symbol}

    # Merge any custom parameters
    if strategy_params:
        strat_params.update(strategy_params)

    # Add strategy (production IbsStrategy class!)
    logger.info(f"Adding IbsStrategy with params: {strat_params}")
    cerebro.addstrategy(IbsStrategy, **strat_params)

    # Add position sizer
    cerebro.addsizer(PercentageSizer, percent=100)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    # Print starting portfolio value
    start_value = cerebro.broker.getvalue()
    logger.info(f'Starting Portfolio Value: ${start_value:,.2f}')

    # Run backtest
    logger.info("Running backtest...")
    results = cerebro.run()
    strat = results[0]

    # Print final portfolio value
    end_value = cerebro.broker.getvalue()
    logger.info(f'Ending Portfolio Value: ${end_value:,.2f}')
    logger.info(f'Total Return: ${end_value - start_value:,.2f} ({((end_value - start_value) / start_value) * 100:.2f}%)')

    # Print analyzer results
    print("\n" + "="*80)
    print(f"BACKTEST RESULTS: {symbol} ({start_date} to {end_date})")
    print("="*80)

    # Sharpe Ratio
    sharpe = strat.analyzers.sharpe.get_analysis()
    print(f"\nSharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")

    # Drawdown
    drawdown = strat.analyzers.drawdown.get_analysis()
    print(f"Max Drawdown: {drawdown.get('max', {}).get('drawdown', 'N/A'):.2f}%")

    # Returns
    returns = strat.analyzers.returns.get_analysis()
    print(f"Total Return: {returns.get('rtot', 0) * 100:.2f}%")

    # Trade Analysis
    trades = strat.analyzers.trades.get_analysis()
    total_trades = trades.get('total', {}).get('total', 0)
    won_trades = trades.get('won', {}).get('total', 0)
    lost_trades = trades.get('lost', {}).get('total', 0)

    print(f"\nTotal Trades: {total_trades}")
    if total_trades > 0:
        win_rate = (won_trades / total_trades) * 100
        print(f"Win Rate: {win_rate:.2f}% ({won_trades} wins, {lost_trades} losses)")

        if won_trades > 0:
            avg_win = trades.get('won', {}).get('pnl', {}).get('average', 0)
            print(f"Average Win: ${avg_win:,.2f}")

        if lost_trades > 0:
            avg_loss = trades.get('lost', {}).get('pnl', {}).get('average', 0)
            print(f"Average Loss: ${avg_loss:,.2f}")

    print("="*80 + "\n")

    return {
        'strategy': strat,
        'start_value': start_value,
        'end_value': end_value,
        'total_return': end_value - start_value,
        'sharpe': sharpe.get('sharperatio'),
        'max_drawdown': drawdown.get('max', {}).get('drawdown'),
        'total_trades': total_trades,
        'win_rate': (won_trades / total_trades) * 100 if total_trades > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description='Backtest runner using production IbsStrategy')
    parser.add_argument('--symbol', type=str, help='Symbol to backtest (e.g., ES)')
    parser.add_argument('--symbols', nargs='+', help='Multiple symbols to backtest')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD, defaults to today)')
    parser.add_argument('--data-dir', type=str, default='data/resampled', help='Data directory')
    parser.add_argument('--cash', type=float, default=100000.0, help='Initial capital')
    parser.add_argument('--commission', type=float, default=COMMISSION_PER_SIDE, help='Commission per side')
    parser.add_argument('--no-ml', action='store_true', help='Disable ML model loading (use default params)')

    args = parser.parse_args()

    # Determine which symbols to backtest
    symbols = []
    if args.symbol:
        symbols.append(args.symbol)
    if args.symbols:
        symbols.extend(args.symbols)

    if not symbols:
        parser.print_help()
        logger.error("\nError: Must specify --symbol or --symbols")
        return

    # Default end date to today if not specified
    end_date = args.end or datetime.now().strftime('%Y-%m-%d')

    # Run backtests
    results = {}
    for symbol in symbols:
        try:
            result = run_backtest(
                symbol=symbol,
                start_date=args.start,
                end_date=end_date,
                data_dir=args.data_dir,
                initial_cash=args.cash,
                commission=args.commission,
                use_ml=not args.no_ml,
            )
            results[symbol] = result
        except Exception as e:
            logger.error(f"Error backtesting {symbol}: {e}", exc_info=True)

    # Summary if multiple symbols
    if len(symbols) > 1:
        print("\n" + "="*80)
        print("SUMMARY - ALL SYMBOLS")
        print("="*80)
        for symbol, result in results.items():
            print(f"{symbol:6s} | Return: {result['total_return']:>10,.2f} | "
                  f"Sharpe: {result['sharpe'] or 'N/A':>6} | "
                  f"Trades: {result['total_trades']:>4} | "
                  f"Win%: {result['win_rate']:>5.1f}%")
        print("="*80 + "\n")


if __name__ == '__main__':
    main()
