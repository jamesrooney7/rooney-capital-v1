#!/usr/bin/env python3
"""
Generate detailed backtest trade data for portfolio optimization.

This script:
1. Loads trained ML models from src/models/
2. Runs backtests for each symbol (2023-2024)
3. Saves detailed trade CSVs to results/ directory with ML feature values
4. Output format matches what portfolio_optimizer_greedy_train_test.py expects

Usage:
    # Generate backtest data for all symbols
    python research/generate_portfolio_backtest_data.py --start 2023-01-01 --end 2024-12-31

    # Specific symbols only
    python research/generate_portfolio_backtest_data.py --symbols ES NQ 6A --start 2023-01-01 --end 2024-12-31
    
    # With custom capital
    python research/generate_portfolio_backtest_data.py --symbols ES --start 2023-01-01 --end 2023-12-31 --initial-cash 1000000000
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

import backtrader as bt
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from research.utils.data_loader import setup_cerebro_with_data, load_symbol_data
from strategy.ibs_strategy import IbsStrategy
from models.loader import load_model_bundle
from strategy.contract_specs import CONTRACT_SPECS, point_value
from config import COMMISSION_PER_SIDE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradeLoggingStrategy(IbsStrategy):
    """IbsStrategy wrapper that logs every trade to CSV with ML feature values."""

    def __init__(self, *args, **kwargs):
        self.output_dir = kwargs.pop('output_dir', None)
        self.filter_start_date = kwargs.pop('filter_start_date', None)
        self.trade_entries = {}  # {order_id: entry_data}
        self.all_trades = []  # Store all trades in memory
        self.warmup_trades_filtered = 0  # Count trades filtered during warmup

        super().__init__(*args, **kwargs)

        # CRITICAL: Force calculation of ALL features (not just enabled ones)
        # This populates ml_feature_param_keys to tell collect_filter_values()
        # to calculate every possible feature, matching how training data was extracted
        self.ml_feature_param_keys = self._get_all_filter_param_keys()
        logger.info(f"Forcing calculation of {len(self.ml_feature_param_keys)} filter parameters for complete feature set")

    def _get_all_filter_param_keys(self) -> set:
        """
        Return comprehensive set of all filter parameter keys.

        This ensures collect_filter_values() calculates ALL possible features,
        not just the ones explicitly enabled in the strategy config.
        This matches how training data was extracted in extract_training_data.py
        """
        keys = {
            # Calendar filters
            'allowedDOW', 'allowedMon', 'enableDOM', 'domDay',
            'enableBegWeek', 'enableEvenOdd',

            # Price/return filters
            'enablePrevDayPct', 'prev_day_pct',
            'enablePrevBarPct', 'prev_bar_pct',

            # IBS filters
            'enableIBSEntry', 'enableIBSExit', 'ibs',
            'enableDailyIBS', 'daily_ibs',
            'enablePrevIBS', 'prev_ibs',
            'enablePrevIBSDaily', 'prev_daily_ibs',

            # Pair filters
            'enablePairIBS', 'pair_ibs',
            'enablePairZ', 'pair_z',

            # RSI filters
            'enableRSIEntry', 'rsi_entry',
            'enableRSIEntry2Len', 'rsi_entry2_len',
            'enableRSIEntry14Len', 'rsi_entry14_len',
            'enableRSIEntry2', 'rsi2_entry',
            'enableDailyRSI', 'daily_rsi',
            'enableDailyRSI2Len', 'daily_rsi2_len',
            'enableDailyRSI14Len', 'daily_rsi14_len',

            # Bollinger Bands
            'enableBBHigh', 'bb_high',
            'enableBBHighD', 'bb_high_d',

            # EMA filters
            'enableEMA8', 'ema8',
            'enableEMA20', 'ema20',

            # ATR filters
            'enableATRZ', 'atrz',
            'enableHourlyATRPercentile', 'hourly_atr_percentile',

            # Volume filters
            'enableVolZ', 'volz',

            # Momentum/distance filters
            'enableDistZ', 'distz',
            'enableMom3Z', 'mom3z',
            'enablePriceZ', 'pricez',

            # Daily ATR/volume
            'enableDATRZ', 'datrz',
            'enableDVolZ', 'dvolz',

            # Trend/ratio filters
            'enableTRATRRatio', 'tratr_ratio',

            # Supply zone
            'use_supply_zone', 'supply_zone',

            # N7 bar
            'enableN7Bar', 'n7_bar',

            # Inside bar
            'enableInsideBar', 'inside_bar',

            # Bear count
            'enableBearCount', 'bear_count',

            # Spiral ER
            'enableSER', 'ser',

            # TWRC
            'enableTWRC', 'twrc',

            # VIX regime
            'enableVIXReg', 'vix_reg',
        }

        # Add cross-asset z-score and return keys dynamically
        cross_symbols = ['ES', 'NQ', 'RTY', 'YM', 'GC', 'SI', 'HG', 'CL', 'NG', 'PL',
                         '6A', '6B', '6C', '6E', '6J', '6M', '6N', '6S', 'TLT', 'VIX']
        timeframes = ['Hour', 'Day']

        for symbol in cross_symbols:
            for tf in timeframes:
                # Z-score filters
                keys.add(f'enable{symbol}ZScore{tf}')
                keys.add(f'{symbol.lower()}_z_score_{tf.lower()}')

                # Daily return filters
                keys.add(f'{symbol.lower()}_daily_return')
                keys.add(f'{symbol.lower()}_hourly_return')

        return keys

    def notify_order(self, order):
        """Capture entry and exit details with ML feature values."""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                # Entry - capture ML features from order.info
                entry_time = bt.num2date(self.hourly.datetime[0])
                entry_price = order.executed.price
                filter_snapshot = order.info.get('filter_snapshot', {})

                self.trade_entries[id(order)] = {
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'filter_snapshot': filter_snapshot,  # Save ML features!
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
                        filter_dt = pd.to_datetime(self.filter_start_date)
                        entry_dt = pd.to_datetime(entry_data['entry_time'])

                        if entry_dt < filter_dt:
                            # This trade was during warmup - don't save it
                            self.warmup_trades_filtered += 1
                            logger.debug(f"Filtered warmup trade: {entry_data['entry_time']} < {self.filter_start_date}")
                            return  # Skip this trade

                    # Save trade with basic info
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

                    # Add all ML feature values from filter_snapshot
                    filter_snapshot = entry_data.get('filter_snapshot', {})
                    if filter_snapshot:
                        # Flatten the filter_snapshot dictionary into the trade_record
                        for key, value in filter_snapshot.items():
                            if key not in trade_record:  # Don't overwrite existing columns
                                trade_record[key] = value

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

            # Save metadata
            metadata = {
                'symbol': self.p.symbol,
                'start_date': self.filter_start_date or "N/A",
                'end_date': "N/A",
                'n_trades': len(self.all_trades),
                'threshold': self.p.ml_threshold if hasattr(self.p, 'ml_threshold') else None,
                'features': list(self.p.ml_features) if hasattr(self.p, 'ml_features') else [],
                'Sharpe_OOS_CPCV': 0,  # Placeholder
                'Profit_Factor': 1.0,  # Placeholder
            }
            
            metadata_path = output_dir / "best.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"✅ Saved metadata to: {metadata_path}")

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
    """Run backtest and save detailed trade data with ML features.

    Args:
        warmup_days: Number of days before start_date to load for warmup (default: 252 = 1 year)
    """
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
    cerebro.broker.setcommission(commission=COMMISSION_PER_SIDE)

    # Set slippage
    spec = CONTRACT_SPECS.get(symbol.upper(), {"tick_size": 0.25})
    tick_size = spec["tick_size"]
    cerebro.broker.set_slippage_fixed(tick_size)

    # Enable cheat-on-close to match original test methodology
    # WARNING: This introduces look-ahead bias but matches how the model was tested
    # Signal at bar close → Execute at bar close (unrealistic, for testing only)
    cerebro.broker.set_coc(True)

    # Load data (primary + references)
    primary_symbol = symbol
    reference_symbols = ['TLT', 'VIX', 'ES', 'NQ', 'RTY', 'YM',
                         'GC', 'SI', 'HG', 'CL', 'NG', 'PL',
                         '6A', '6B', '6C', '6E', '6J', '6M', '6N', '6S']
    reference_symbols = [s for s in reference_symbols if s != primary_symbol]

    symbols_to_load = [primary_symbol]

    # Try loading reference symbols from warmup start date
    for ref_symbol in reference_symbols:
        try:
            hourly_df, daily_df = load_symbol_data(
                ref_symbol,
                data_dir=data_dir,
                start_date=warmup_start_date,
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
        start_date=warmup_start_date,
        end_date=end_date
    )

    # Get strategy kwargs from model bundle
    strat_kwargs = bundle.strategy_kwargs()
    strat_kwargs['symbol'] = symbol
    strat_kwargs['output_dir'] = output_dir / f"{symbol}_optimization"
    strat_kwargs['filter_start_date'] = start_date  # Exclude warmup trades

    # Add strategy
    cerebro.addstrategy(TradeLoggingStrategy, **strat_kwargs)

    # Run backtest
    logger.info("Running backtest...")
    cerebro.run()

    return {
        'symbol': symbol,
        'n_trades': 'See CSV',
    }


def main():
    parser = argparse.ArgumentParser(description='Generate backtest trade data with ML features')
    parser.add_argument('--symbols', nargs='+', required=True, help='Symbols to backtest')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--data-dir', type=str, default='data/resampled', help='Data directory')
    parser.add_argument('--initial-cash', type=float, default=100000.0, help='Initial capital')

    args = parser.parse_args()

    output_dir = Path(args.output)
    symbols = args.symbols

    logger.info("\n" + "="*80)
    logger.info("# GENERATING BACKTEST TRADE DATA")
    logger.info("="*80)
    logger.info(f"Period: {args.start} to {args.end}")
    logger.info(f"Symbols: {len(symbols)} - {', '.join(symbols)}")
    logger.info(f"Output: {output_dir}/")
    logger.info(f"Mode: SEQUENTIAL")
    logger.info("="*80)
    logger.info("")

    results = {}
    failed_symbols = []

    # Sequential execution
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
            else:
                failed_symbols.append(symbol)

        except Exception as e:
            logger.error(f"❌ {symbol}: Failed - {e}")
            failed_symbols.append(symbol)
            import traceback
            traceback.print_exc()

    # Summary
    logger.info("\n" + "="*80)
    logger.info(f"COMPLETED: {len(results)}/{len(symbols)} symbols")
    if failed_symbols:
        logger.info(f"FAILED: {', '.join(failed_symbols)}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
