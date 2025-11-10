#!/usr/bin/env python3
"""
Generate detailed backtest trade data for portfolio optimization.

This script:
1. Loads trained ML models from src/models/
2. Runs backtests for each symbol (2023-2024)
3. Saves detailed trade CSVs to results/ directory
4. Output format matches what portfolio_optimizer_greedy_train_test.py expects

Usage:
    # Generate backtest data for all symbols
    python research/generate_portfolio_backtest_data.py --start 2023-01-01 --end 2024-12-31

    # Specific symbols only
    python research/generate_portfolio_backtest_data.py --symbols ES NQ 6A --start 2023-01-01 --end 2024-12-31
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import backtrader as bt
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from research.utils.data_loader import setup_cerebro_with_data
from strategy.ibs_strategy import IbsStrategy
from models.loader import load_model_bundle
from strategy.contract_specs import CONTRACT_SPECS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradeLoggingStrategy(IbsStrategy):
    """IbsStrategy wrapper that logs every trade to CSV."""

    def __init__(self, *args, **kwargs):
        self.output_dir = kwargs.pop('output_dir', None)
        self.filter_start_date = kwargs.pop('filter_start_date', None)
        self.csv_file = None
        self.csv_writer = None
        self.trade_entries = {}  # {order_id: entry_data}
        self.all_trades = []  # Store all trades in memory
        self.warmup_trades_filtered = 0  # Count trades filtered during warmup

        super().__init__(*args, **kwargs)

    def notify_order(self, order):
        """Capture entry and exit details."""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                # Entry
                entry_time = bt.num2date(self.hourly.datetime[0])
                entry_price = order.executed.price

                self.trade_entries[id(order)] = {
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                }

            elif order.issell():
                # Exit
                exit_time = bt.num2date(self.hourly.datetime[0])
                exit_price = order.executed.price

                # Find matching entry
                if self.trade_entries:
                    entry_id = max(self.trade_entries.keys())
                    entry_data = self.trade_entries.pop(entry_id)

                    # Calculate PnL
                    entry_price = entry_data['entry_price']
                    price_return = (exit_price - entry_price) / entry_price

                    # Get contract specs for PnL
                    from strategy.contract_specs import point_value
                    pv = point_value(self.p.symbol)

                    # Gross PnL
                    gross_pnl = (exit_price - entry_price) * pv

                    # Net PnL (after commissions: $1 per side = $2 total)
                    commission_total = 2.00
                    pnl_usd = gross_pnl - commission_total

                    # ML selected? (Always 1 since this strategy uses ML filtering)
                    model_selected = 1

                    # Check if trade is within the target date range (exclude warmup)
                    if self.filter_start_date:
                        import pandas as pd
                        filter_dt = pd.to_datetime(self.filter_start_date)
                        entry_dt = pd.to_datetime(entry_data['entry_time'])

                        if entry_dt < filter_dt:
                            # This trade was during warmup - don't save it
                            self.warmup_trades_filtered += 1
                            logger.debug(f"Filtered warmup trade: {entry_data['entry_time']} < {self.filter_start_date}")
                            return  # Skip this trade

                    # Save trade
                    trade_record = {
                        'Date/Time': entry_data['entry_time'],
                        'Exit Date/Time': exit_time,
                        'Entry_Price': entry_price,
                        'Exit_Price': exit_price,
                        'y_return': price_return,
                        'y_binary': 1 if pnl_usd > 0 else 0,
                        'y_pnl_usd': pnl_usd,
                        'y_pnl_gross': gross_pnl,
                        'Model_Selected': model_selected,
                        'pnl_usd': pnl_usd,
                    }

                    self.all_trades.append(trade_record)

                    logger.debug(f"Trade: {entry_data['entry_time']} -> {exit_time} | PnL=${pnl_usd:.2f}")

        # Call parent
        super().notify_order(order)

    def stop(self):
        """Save all trades to CSV when backtest completes."""
        if self.output_dir and self.all_trades:
            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            csv_path = output_dir / f"{self.p.symbol}_rf_best_trades.csv"

            df = pd.DataFrame(self.all_trades)
            df.to_csv(csv_path, index=False)

            logger.info(f"✅ Saved {len(self.all_trades)} trades to: {csv_path}")
            if self.warmup_trades_filtered > 0:
                logger.info(f"   (Filtered {self.warmup_trades_filtered} warmup trades)")

        super().stop()


def run_backtest_with_trade_logging(
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
    data_dir: str = 'data/resampled',
    initial_cash: float = 100000.0,
    warmup_days: int = 252,  # 1 year warmup
):
    """Run backtest and save detailed trade data.

    Args:
        warmup_days: Number of days before start_date to load for warmup (default: 252 = 1 year)
    """
    from datetime import datetime, timedelta

    # Calculate warmup start date
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    warmup_start_dt = start_dt - timedelta(days=warmup_days)
    warmup_start_date = warmup_start_dt.strftime('%Y-%m-%d')

    logger.info(f"\n{'='*80}")
    logger.info(f"Backtesting {symbol}: {start_date} to {end_date}")
    logger.info(f"Warmup period: {warmup_start_date} to {start_date} ({warmup_days} days)")
    logger.info(f"{'='*80}")

    # Load ML model
    try:
        bundle = load_model_bundle(symbol, base_dir='src/models')
        threshold_str = f"{bundle.threshold:.3f}" if bundle.threshold is not None else "N/A"
        logger.info(f"✅ Loaded ML model: {len(bundle.features)} features, threshold={threshold_str}")
    except Exception as e:
        logger.error(f"❌ Failed to load model for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Create Cerebro
    cerebro = bt.Cerebro(runonce=False)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=1.00)  # $1 per side

    # Set slippage
    spec = CONTRACT_SPECS.get(symbol.upper(), {"tick_size": 0.25})
    tick_size = spec["tick_size"]
    cerebro.broker.set_slippage_fixed(tick_size)

    # Load data (primary + references)
    primary_symbol = symbol
    reference_symbols = ['TLT', 'VIX', 'ES', 'NQ', 'RTY', 'YM',
                         'GC', 'SI', 'HG', 'CL', 'NG', 'PL',
                         '6A', '6B', '6C', '6E', '6J', '6M', '6N', '6S']
    reference_symbols = [s for s in reference_symbols if s != primary_symbol]

    symbols_to_load = [primary_symbol]

    # Try loading reference symbols
    from research.utils.data_loader import load_symbol_data
    for ref_symbol in reference_symbols:
        try:
            hourly_df, daily_df = load_symbol_data(
                ref_symbol,
                data_dir=data_dir,
                start_date=start_date,
                end_date=end_date
            )

            if len(hourly_df) >= 252 * 6 and len(daily_df) >= 252:
                symbols_to_load.append(ref_symbol)
        except:
            pass

    logger.info(f"Loading {len(symbols_to_load)} data feeds ({primary_symbol} + {len(symbols_to_load)-1} references)")

    # Load data with warmup period
    setup_cerebro_with_data(
        cerebro,
        symbols=symbols_to_load,
        data_dir=data_dir,
        start_date=warmup_start_date,  # Use warmup start
        end_date=end_date
    )

    # Add strategy with ML model
    # Pass actual start_date to filter out warmup trades
    symbol_output_dir = output_dir / f"{symbol}_optimization"

    cerebro.addstrategy(
        TradeLoggingStrategy,
        symbol=symbol,
        output_dir=str(symbol_output_dir),
        ml_model=bundle.model,
        ml_features=bundle.features,
        ml_threshold=bundle.threshold,
        filter_start_date=start_date,  # Only save trades from this date forward
    )

    # Run backtest
    logger.info("Running backtest...")
    results = cerebro.run()
    strat = results[0]

    # Save metadata
    metadata = {
        'symbol': symbol,
        'start_date': start_date,
        'end_date': end_date,
        'n_trades': len(strat.all_trades),
        'threshold': bundle.threshold,
        'features': bundle.features,
        'Sharpe_OOS_CPCV': bundle.metadata.get('test_metrics', {}).get('sharpe', 0) if hasattr(bundle, 'metadata') else 0,
        'Profit_Factor': bundle.metadata.get('test_metrics', {}).get('profit_factor', 1.0) if hasattr(bundle, 'metadata') else 1.0,
    }

    symbol_output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = symbol_output_dir / "best.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"✅ Saved metadata to: {metadata_path}")

    # Calculate summary
    if strat.all_trades:
        df = pd.DataFrame(strat.all_trades)
        total_pnl = df['pnl_usd'].sum()
        win_rate = (df['y_binary'] == 1).mean()

        logger.info(f"Summary: {len(df)} trades | Total PnL: ${total_pnl:,.2f} | Win Rate: {win_rate*100:.1f}%")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Generate detailed backtest trade data for portfolio optimization'
    )

    parser.add_argument('--symbols', nargs='+',
                       help='Symbols to backtest (default: all 18)')
    parser.add_argument('--start', type=str, required=True,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--data-dir', type=str, default='data/resampled',
                       help='Data directory')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for trade CSVs')
    parser.add_argument('--initial-cash', type=float, default=100000.0,
                       help='Initial capital per backtest')
    parser.add_argument('--parallel', action='store_true',
                       help='Run backtests in parallel (faster)')
    parser.add_argument('--jobs', type=int, default=4,
                       help='Number of parallel jobs (default: 4)')

    args = parser.parse_args()

    # Default to all 18 symbols
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = ['6A', '6B', '6C', '6E', '6J', '6M', '6N', '6S',
                   'ES', 'NQ', 'RTY', 'YM',
                   'GC', 'SI', 'HG', 'PL',
                   'CL', 'NG']

    output_dir = Path(args.output_dir)

    logger.info(f"\n{'#'*80}")
    logger.info(f"# GENERATING BACKTEST TRADE DATA")
    logger.info(f"{'#'*80}")
    logger.info(f"Period: {args.start} to {args.end}")
    logger.info(f"Symbols: {len(symbols)} - {', '.join(symbols)}")
    logger.info(f"Output: {output_dir}/")
    if args.parallel:
        logger.info(f"Mode: PARALLEL ({args.jobs} jobs)")
    else:
        logger.info(f"Mode: SEQUENTIAL")
    logger.info(f"{'#'*80}\n")

    results = {}
    success_count = 0
    failed_symbols = []

    if args.parallel:
        # Parallel execution using multiprocessing
        from multiprocessing import Pool, cpu_count
        from functools import partial

        # Limit jobs to available CPUs
        num_jobs = min(args.jobs, cpu_count(), len(symbols))
        logger.info(f"Running {num_jobs} parallel jobs (CPU count: {cpu_count()})\n")

        # Create a partial function with fixed parameters
        run_func = partial(
            run_backtest_with_trade_logging,
            start_date=args.start,
            end_date=args.end,
            output_dir=output_dir,
            data_dir=args.data_dir,
            initial_cash=args.initial_cash,
        )

        # Run in parallel
        with Pool(processes=num_jobs) as pool:
            results_list = pool.map(run_func, symbols)

        # Collect results
        for symbol, metadata in zip(symbols, results_list):
            if metadata:
                results[symbol] = metadata
                success_count += 1
            else:
                failed_symbols.append(symbol)

    else:
        # Sequential execution (original behavior)
        for symbol in symbols:
            try:
                metadata = run_backtest_with_trade_logging(
                    symbol=symbol,
                    start_date=args.start,
                    end_date=args.end,
                    output_dir=output_dir,
                    data_dir=args.data_dir,
                    initial_cash=args.initial_cash,
                )

                if metadata:
                    results[symbol] = metadata
                    success_count += 1
                else:
                    failed_symbols.append(symbol)

            except Exception as e:
                logger.error(f"❌ {symbol}: Failed - {e}")
                failed_symbols.append(symbol)
                import traceback
                traceback.print_exc()

    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info(f"BACKTEST DATA GENERATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"✅ Success: {success_count}/{len(symbols)}")

    if failed_symbols:
        logger.warning(f"❌ Failed: {', '.join(failed_symbols)}")

    logger.info(f"\nOutput saved to: {output_dir}/")
    logger.info(f"\nNext step: Run portfolio optimizer:")
    logger.info(f"  python research/portfolio_optimizer_greedy_train_test.py \\")
    logger.info(f"    --results-dir {output_dir} \\")
    logger.info(f"    --train-start 2023-01-01 --train-end 2023-12-31 \\")
    logger.info(f"    --test-start 2024-01-01 --test-end 2024-12-31 \\")
    logger.info(f"    --max-dd-limit 5000 \\")
    logger.info(f"    --update-config")
    logger.info("")

    return 0 if success_count == len(symbols) else 1


if __name__ == '__main__':
    sys.exit(main())
