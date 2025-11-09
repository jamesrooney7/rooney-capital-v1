"""
Strategy Worker - Main entry point for running a single strategy.

This is the core process that runs one strategy independently:
- Loads strategy configuration
- Connects to Redis data hub for market data
- Creates Backtrader Cerebro with Redis feeds
- Loads and runs the strategy
- Routes orders to TradersPost
- Writes heartbeat files for monitoring
- Handles graceful shutdown

Usage:
    python -m src.runner.strategy_worker --strategy ibs --config config.multi_alpha.yml
"""

import argparse
import gc
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import backtrader as bt
import pandas as pd

# Import configuration
from src.config import load_config, RuntimeConfig, StrategyConfig

# Import feeds
from src.feeds import RedisLiveData, RedisResampledData

# Import Bar dataclass from legacy databento_bridge
from src.runner.databento_bridge import Bar

# Import contract map for resolving active contracts
from src.runner.contract_map import ContractMap, load_contract_map

# Import strategy factory
from src.strategy.strategy_factory import load_strategy, create_strategy_config

# Import portfolio coordinator (will adapt for multi-alpha)
from src.runner.portfolio_coordinator import PortfolioCoordinator

# Import TradersPost client (will adapt for multi-alpha)
from src.runner.traderspost_client import TradersPostClient

# Import ML model loader
try:
    from src.models.loader import load_model_bundle
except ImportError:
    load_model_bundle = None

logger = logging.getLogger(__name__)


class StrategyWorker:
    """
    Worker process that runs a single strategy.

    Responsibilities:
    - Load strategy configuration
    - Connect to Redis data hub
    - Set up Backtrader with Redis feeds
    - Load ML models
    - Run strategy event loop
    - Handle graceful shutdown
    """

    def __init__(
        self,
        strategy_name: str,
        config: RuntimeConfig,
        log_level: str = "INFO"
    ):
        """
        Initialize strategy worker.

        Args:
            strategy_name: Name of strategy to run (e.g., "ibs")
            config: Complete runtime configuration
            log_level: Logging level
        """
        self.strategy_name = strategy_name
        self.config = config
        self.log_level = log_level

        # Get strategy-specific config
        self.strategy_config = config.get_strategy(strategy_name)
        if not self.strategy_config:
            raise ValueError(f"Strategy '{strategy_name}' not found in config")

        if not self.strategy_config.enabled:
            raise ValueError(f"Strategy '{strategy_name}' is disabled in config")

        # Backtrader instance
        self.cerebro: Optional[bt.Cerebro] = None

        # Components
        self.portfolio_coordinator: Optional[PortfolioCoordinator] = None
        self.traderspost_client: Optional[TradersPostClient] = None

        # ML models per instrument
        self.ml_models: Dict[str, Any] = {}

        # Data feeds (store references for warmup)
        self.data_feeds: Dict[str, Any] = {}  # feed_name -> feed object

        # Heartbeat
        self.heartbeat_file = self.strategy_config.heartbeat_file
        self.heartbeat_interval = self.strategy_config.heartbeat_interval
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # State
        self.start_time: Optional[datetime] = None
        self.running = False

        # Warmup batching configuration (matches legacy system)
        self.warmup_batch_size = getattr(self.strategy_config, 'historical_warmup_batch_size', 5000)
        self.warmup_queue_soft_limit = getattr(self.strategy_config, 'historical_warmup_queue_soft_limit', 20000)
        self.warmup_wait_log_interval = 5.0  # Log every 5 seconds when waiting

        # Load contract map for resolving symbols to active contracts
        # This prevents downloading ALL contracts when requesting historical data
        contract_map_path = getattr(config, 'contract_map_path', 'Data/Databento_contract_map.yml')
        self.contract_map: Optional[ContractMap] = None
        try:
            if Path(contract_map_path).exists():
                self.contract_map = load_contract_map(contract_map_path)
                logger.info(f"Loaded contract map from {contract_map_path}")
            else:
                logger.warning(f"Contract map not found at {contract_map_path}, will use parent symbols")
        except Exception as e:
            logger.warning(f"Failed to load contract map: {e}, will use parent symbols")

        logger.info(
            f"StrategyWorker initialized for '{strategy_name}' "
            f"(instruments={len(self.strategy_config.instruments)})"
        )

    def setup(self):
        """Set up all components before running."""
        logger.info(f"Setting up strategy worker for '{self.strategy_name}'...")

        # Create portfolio coordinator
        self._setup_portfolio_coordinator()

        # Create TradersPost client
        self._setup_traderspost_client()

        # Load ML models
        self._load_ml_models()

        # Create Backtrader Cerebro
        self._setup_cerebro()

        # Load historical data for warmup
        if self.strategy_config.load_historical_warmup:
            self._load_historical_warmup()

        logger.info("Strategy worker setup complete")

    def _setup_portfolio_coordinator(self):
        """Create portfolio coordinator for this strategy."""
        self.portfolio_coordinator = PortfolioCoordinator(
            max_positions=self.strategy_config.max_positions,
            daily_stop_loss=self.strategy_config.daily_stop_loss,
            emergency_exit_callback=self._on_emergency_exit,
            daily_summary_callback=self._on_daily_summary
        )

        logger.info(
            f"Portfolio coordinator: max_positions={self.strategy_config.max_positions}, "
            f"daily_stop_loss=${self.strategy_config.daily_stop_loss}"
        )

    def _setup_traderspost_client(self):
        """Create TradersPost client for this strategy."""
        webhook_url = self.strategy_config.broker_account

        if not webhook_url:
            logger.warning("No broker_account (webhook) configured - orders will not be sent")

        self.traderspost_client = TradersPostClient(
            webhook_url=webhook_url,
            max_retries=3,
            backoff_factor=0.5,
            timeout=10.0
        )

        logger.info(f"TradersPost client configured: {webhook_url[:50]}..." if webhook_url else "TradersPost client: NO WEBHOOK")

    def _load_ml_models(self):
        """Load ML models for each instrument."""
        if not load_model_bundle:
            logger.warning("ML model loader not available, skipping model loading")
            return

        models_path = Path(self.strategy_config.models_path)

        if not models_path.exists():
            logger.warning(f"Models directory not found: {models_path}")
            return

        for symbol in self.strategy_config.instruments:
            try:
                bundle = load_model_bundle(symbol, base_dir=str(models_path))
                self.ml_models[symbol] = bundle
                logger.info(
                    f"Loaded ML model for {symbol}: "
                    f"{len(bundle.features)} features, threshold={bundle.threshold}"
                )
            except Exception as e:
                logger.warning(f"Failed to load ML model for {symbol}: {e}")

        logger.info(f"Loaded {len(self.ml_models)}/{len(self.strategy_config.instruments)} ML models")

    def _setup_cerebro(self):
        """Create and configure Backtrader Cerebro instance."""
        self.cerebro = bt.Cerebro()

        # Set broker cash
        self.cerebro.broker.setcash(self.strategy_config.starting_cash)

        # Add data feeds for each instrument
        self._add_data_feeds()

        # Add strategy
        self._add_strategy()

        logger.info("Cerebro configured successfully")

    def _add_data_feeds(self):
        """Add Redis data feeds for all instruments + reference feeds."""
        # Add full feeds (minute/hour/day) for this strategy's instruments
        for symbol in self.strategy_config.instruments:
            # Get instrument config
            instr_config = self.config.get_instrument(symbol)
            if not instr_config:
                logger.warning(f"No instrument config for {symbol}, skipping")
                continue

            # Add minute feed
            data_minute = RedisLiveData(
                symbol=symbol,
                redis_host=self.strategy_config.redis_host,
                redis_port=self.strategy_config.redis_port,
                name=f"{symbol}_minute"
            )
            self.cerebro.adddata(data_minute, name=f"{symbol}_minute")
            self.data_feeds[f"{symbol}_minute"] = data_minute

            # Add hourly resampled feed (aggregates from minute feed)
            data_hourly = RedisResampledData(
                symbol=symbol,
                source_feed=data_minute,  # Aggregate from minute feed
                bar_interval_minutes=60,  # Hourly bars
                session_end_hour=23,
                name=f"{symbol}_hour"
            )
            self.cerebro.adddata(data_hourly, name=f"{symbol}_hour")
            self.data_feeds[f"{symbol}_hour"] = data_hourly

            # Add daily resampled feed (aggregates from minute feed)
            data_daily = RedisResampledData(
                symbol=symbol,
                source_feed=data_minute,  # Aggregate from minute feed
                bar_interval_minutes=1440,  # Daily bars
                session_end_hour=23,  # CME session end
                name=f"{symbol}_day"
            )
            self.cerebro.adddata(data_daily, name=f"{symbol}_day")
            self.data_feeds[f"{symbol}_day"] = data_daily

            logger.info(f"Added Redis feeds for {symbol} (minute/hourly/daily)")

        # Add reference feeds (hour/day only) for ALL other configured instruments
        # This allows cross-instrument features to work
        # Note: We need minute feeds for these too, since resampled feeds aggregate from minute
        for symbol in self.config.instruments.keys():
            if symbol in self.strategy_config.instruments:
                continue  # Already added above
            if symbol == "ZB":
                continue  # Handled separately below

            # Add minute reference feed (needed for resampling)
            data_minute_ref = RedisLiveData(
                symbol=symbol,
                redis_host=self.strategy_config.redis_host,
                redis_port=self.strategy_config.redis_port,
                name=f"{symbol}_minute"
            )
            self.cerebro.adddata(data_minute_ref, name=f"{symbol}_minute")
            self.data_feeds[f"{symbol}_minute"] = data_minute_ref

            # Add hourly reference feed (aggregates from minute)
            data_hourly_ref = RedisResampledData(
                symbol=symbol,
                source_feed=data_minute_ref,
                bar_interval_minutes=60,
                session_end_hour=23,
                name=f"{symbol}_hour"
            )
            self.cerebro.adddata(data_hourly_ref, name=f"{symbol}_hour")
            self.data_feeds[f"{symbol}_hour"] = data_hourly_ref

            # Add daily reference feed (aggregates from minute)
            data_daily_ref = RedisResampledData(
                symbol=symbol,
                source_feed=data_minute_ref,
                bar_interval_minutes=1440,
                session_end_hour=23,
                name=f"{symbol}_day"
            )
            self.cerebro.adddata(data_daily_ref, name=f"{symbol}_day")
            self.data_feeds[f"{symbol}_day"] = data_daily_ref

        logger.info(f"Added reference feeds for {len(self.config.instruments) - len(self.strategy_config.instruments)} symbols")

        # Add reference feed: ZB (Treasury futures) as TLT_day
        # The IBS strategy requires a TLT_day feed for market regime detection
        if self.config.get_instrument("ZB"):
            # Add ZB minute feed (needed for resampling)
            data_zb_minute = RedisLiveData(
                symbol="ZB",
                redis_host=self.strategy_config.redis_host,
                redis_port=self.strategy_config.redis_port,
                name="ZB_minute"
            )
            self.cerebro.adddata(data_zb_minute, name="ZB_minute")
            self.data_feeds["ZB_minute"] = data_zb_minute

            # Add ZB daily feed as TLT_day (aggregates from minute)
            data_tlt = RedisResampledData(
                symbol="ZB",
                source_feed=data_zb_minute,
                bar_interval_minutes=1440,
                session_end_hour=23,
                name="TLT_day"
            )
            self.cerebro.adddata(data_tlt, name="TLT_day")
            self.data_feeds["TLT_day"] = data_tlt
            logger.info("Added reference feed: ZB as TLT_day")

            # Add ZB hourly feed (aggregates from minute)
            data_zb_hour = RedisResampledData(
                symbol="ZB",
                source_feed=data_zb_minute,
                bar_interval_minutes=60,
                session_end_hour=23,
                name="ZB_hour"
            )
            self.cerebro.adddata(data_zb_hour, name="ZB_hour")
            self.data_feeds["ZB_hour"] = data_zb_hour
            logger.info("Added reference feed: ZB_hour")

    def _add_strategy(self):
        """Add strategy to Cerebro."""
        # Load strategy class from factory using strategy_type
        strategy_type = getattr(self.strategy_config, 'strategy_type', self.strategy_name)
        strategy_class = load_strategy(strategy_type)

        # Prepare strategy configuration
        # For now, we'll use the first instrument as the primary symbol
        # In a full implementation, strategies might iterate over all instruments
        primary_symbol = self.strategy_config.instruments[0] if self.strategy_config.instruments else "ES"

        # Get ML model for primary symbol
        ml_bundle = self.ml_models.get(primary_symbol)

        # Create strategy params
        strategy_params = create_strategy_config(
            strategy_name=self.strategy_name,
            symbol=primary_symbol,
            config=self.strategy_config.to_dict(),
            portfolio_coordinator=self.portfolio_coordinator,
            ml_model=ml_bundle.model if ml_bundle else None,
            ml_features=ml_bundle.features if ml_bundle else None,
            ml_threshold=ml_bundle.threshold if ml_bundle else None,
            callbacks={
                'on_order': self._on_order,
                'on_trade': self._on_trade
            }
        )

        # Add strategy to Cerebro
        self.cerebro.addstrategy(strategy_class, **strategy_params)

        logger.info(f"Added strategy '{self.strategy_name}' for {primary_symbol}")

    def _convert_databento_to_bars(
        self,
        symbol: str,
        data: Any,
        compression: str = "1min"
    ) -> List[Bar]:
        """
        Convert Databento historical data to Bar objects.

        This matches the legacy _convert_databento_to_bt_bars logic.
        """
        if data is None:
            return []

        # Convert to DataFrame
        if hasattr(data, "to_df"):
            df = data.to_df()
        elif hasattr(data, "to_pandas"):
            df = data.to_pandas()
        else:
            df = pd.DataFrame(data)

        if df is None or df.empty:
            return []

        df = df.copy()

        # Extract timestamps
        if "ts_event" in df.columns:
            timestamps = pd.to_datetime(df["ts_event"], unit="ns", utc=True)
        elif df.index.name == "ts_event":
            timestamps = pd.to_datetime(df.index, unit="ns", utc=True)
        elif isinstance(df.index, pd.DatetimeIndex):
            timestamps = pd.to_datetime(df.index, utc=True)
        else:
            raise ValueError(f"Historical payload for {symbol} missing ts_event column")

        timestamps = pd.DatetimeIndex(timestamps)

        # Map column names
        column_aliases = {
            "open": ("open", "open_px", "open_price"),
            "high": ("high", "high_px", "high_price"),
            "low": ("low", "low_px", "low_price"),
            "close": ("close", "close_px", "px", "price"),
        }

        resolved_columns = {}
        for target, candidates in column_aliases.items():
            for candidate in candidates:
                if candidate in df.columns:
                    resolved_columns[target] = candidate
                    break

        volume_column = next(
            (col for col in ("volume", "size", "qty", "trade_sz") if col in df.columns),
            None,
        )

        # Build OHLCV dataframe
        if set(resolved_columns) == set(column_aliases):
            # Full OHLCV data available
            ohlcv = pd.DataFrame(
                {
                    key: pd.to_numeric(df[col], errors="coerce").to_numpy()
                    for key, col in resolved_columns.items()
                },
                index=timestamps,
            )
            if volume_column:
                ohlcv["volume"] = pd.to_numeric(df[volume_column], errors="coerce").to_numpy()
            else:
                ohlcv["volume"] = 0.0
            ohlcv = ohlcv.sort_index()
            ohlcv = ohlcv.loc[~ohlcv.index.duplicated(keep="last")]
        else:
            # Only price data - need to resample to OHLCV
            price_column = resolved_columns.get("close") or next(
                (col for col in ("price", "px", "close", "trade_px") if col in df.columns),
                None,
            )
            if price_column is None:
                raise ValueError(f"Historical payload for {symbol} missing price column")

            prices = pd.to_numeric(df[price_column], errors="coerce").to_numpy()
            volumes = pd.to_numeric(df[volume_column], errors="coerce").to_numpy() if volume_column else [0.0] * len(df)

            frame = pd.DataFrame(
                {"price": prices, "volume": volumes}, index=timestamps
            )
            frame = frame.sort_index()
            frame = frame.dropna(subset=["price"])
            if frame.empty:
                return []

            # Resample to 1-minute OHLCV
            ohlcv = frame.resample("1min").agg(
                {"price": ["first", "max", "min", "last"], "volume": "sum"}
            )

            if ohlcv.empty:
                return []

            ohlcv.columns = ["open", "high", "low", "close", "volume"]
            ohlcv = ohlcv.sort_index()

        ohlcv = ohlcv.loc[~ohlcv.index.duplicated(keep="last")]

        # Resample to target compression if needed
        if compression != "1min":
            # Use session-aware resampling to match legacy system and live data
            # CME session runs from 23:00 UTC to 23:00 UTC next day
            if compression == "1d":
                # Resample with offset to align to 23:00 UTC (session end)
                # This matches the legacy ResampledLiveData session_end_hour=23 logic
                aggregation = ohlcv.resample("1D", offset="23h").agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
                )
            elif compression == "1h":
                # Hourly resampling aligns to top of hour (standard)
                aggregation = ohlcv.resample("1h").agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
                )
            else:
                # Generic resampling for other compressions
                aggregation = ohlcv.resample(compression).agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
                )

            if aggregation.empty:
                return []
            ohlcv = aggregation

        # Convert to Bar objects
        bars: List[Bar] = []
        last_close: Optional[float] = None

        for timestamp, row in ohlcv.iterrows():
            open_price = row.open
            close_price = row.close
            high_price = row.high
            low_price = row.low
            volume = row.volume

            if pd.notna(open_price) and pd.notna(close_price):
                open_value = float(open_price)
                close_value = float(close_price)
                high_value = float(high_price) if pd.notna(high_price) else close_value
                low_value = float(low_price) if pd.notna(low_price) else close_value
                volume_value = float(volume) if pd.notna(volume) else 0.0
                last_close = close_value
            elif last_close is not None:
                # Fill gaps with last close
                open_value = high_value = low_value = close_value = float(last_close)
                volume_value = 0.0
            else:
                continue

            bars.append(
                Bar(
                    symbol=symbol,
                    timestamp=timestamp.to_pydatetime(),
                    open=open_value,
                    high=high_value,
                    low=low_value,
                    close=close_value,
                    volume=volume_value,
                )
            )

        return bars

    @staticmethod
    def _warmup_backlog_size(feed: Any) -> int:
        """Get the current warmup backlog size for a feed (matches legacy system)."""
        getter = getattr(feed, "warmup_backlog_size", None)
        if callable(getter):
            try:
                return int(getter())
            except Exception:
                logger.debug("Failed to query warmup backlog size", exc_info=True)

        deque_obj = getattr(feed, "_warmup_bars", None)
        if deque_obj is None:
            return 0

        try:
            return len(deque_obj)
        except Exception:
            return 0

    def _wait_for_warmup_capacity(
        self,
        feed: Any,
        *,
        symbol: str,
        incoming: int,
        limit: int,
        pending_batches: int = 1,
        total_batches: Optional[int] = None,
    ) -> None:
        """
        Wait for warmup queue to have capacity before adding more bars.

        This prevents overwhelming the feed's warmup queue and allows Cerebro
        to consume bars while we're still loading (matches legacy system).
        """
        if incoming <= 0:
            return

        base_limit = max(limit, incoming)
        pending_batches = max(int(pending_batches), 1)
        total_batches = max(int(total_batches or pending_batches), pending_batches)
        extra_batches = max(pending_batches - 1, total_batches - 1)

        if incoming:
            effective_limit = base_limit + extra_batches * incoming
        else:
            effective_limit = base_limit

        log_interval = max(float(self.warmup_wait_log_interval), 0.0)
        next_log: Optional[float]
        if log_interval:
            next_log = time.monotonic()
        else:
            next_log = None

        while not self.stop_event.is_set():
            backlog = self._warmup_backlog_size(feed)
            if backlog + incoming <= effective_limit:
                return

            now = time.monotonic()
            if next_log is not None and now >= next_log:
                logger.info(
                    "Waiting for warmup backlog to drain for %s (queued=%d, limit=%d)",
                    symbol,
                    backlog,
                    effective_limit,
                )
                next_log = now + log_interval

            time.sleep(0.05)

        logger.debug("Warmup backlog wait aborted for %s: stop requested", symbol)

    def _load_historical_warmup(self):
        """Load historical data from Databento and populate feeds for indicator warmup."""
        try:
            # Import databento and historical loader
            import databento as db
            from datetime import timedelta, timezone

            logger.info("Loading historical data for warmup...")

            # Get Databento API key from environment
            api_key = os.environ.get('DATABENTO_API_KEY')
            if not api_key:
                logger.warning("DATABENTO_API_KEY not found, skipping warmup")
                return

            # Get dataset from first instrument (assuming all use same dataset)
            dataset = "GLBX.MDP3"  # Default CME dataset
            if self.strategy_config.instruments:
                first_symbol = self.strategy_config.instruments[0]
                instr_config = self.config.get_instrument(first_symbol)
                if instr_config:
                    dataset = getattr(instr_config, 'databento_dataset', 'GLBX.MDP3')

            # Create Databento Historical client
            client = db.Historical(api_key)

            # Collect all symbols we need warmup data for
            all_symbols = set()

            # Traded instruments need all timeframes
            for symbol in self.strategy_config.instruments:
                all_symbols.add(symbol)

            # Reference instruments
            for symbol in self.config.instruments.keys():
                if symbol not in self.strategy_config.instruments:
                    all_symbols.add(symbol)

            logger.info(f"Loading warmup data for {len(all_symbols)} symbols...")

            # Schema preferences for OHLCV historical data
            # Note: Databento uses 'ohlcv-1m' not 'ohlcv-1min'
            schema_preferences = ("ohlcv-1m", "ohlcv-1s", "mbp-1")

            # Load daily bars first (252 trading days)
            # NOTE: We request 1-minute data which the feeds will aggregate to daily
            daily_lookback = self.strategy_config.historical_lookback_days
            end = datetime.now(tz=timezone.utc) - timedelta(hours=24)
            start = end - timedelta(days=daily_lookback)

            logger.info(f"Loading daily warmup data ({daily_lookback} days)...")

            symbols_completed = 0
            for symbol in all_symbols:
                symbols_completed += 1
                logger.info(f"Loading daily warmup for {symbol} ({symbols_completed}/{len(all_symbols)})...")
                try:
                    # Resolve symbol using contract map
                    # For OHLCV data, stype_in="parent" is safe and only returns front month
                    request_symbols: List[str] = []
                    stype_in = "parent"  # Default

                    if self.contract_map is not None:
                        subscription = self.contract_map.subscription_for(symbol)
                        if subscription and subscription.codes:
                            # Use subscription codes and stype_in from contract map
                            request_symbols.extend(subscription.codes)
                            stype_in = subscription.stype_in

                    # Fallback: use product_id from config
                    if not request_symbols:
                        instr_config = self.config.get_instrument(symbol)
                        if instr_config:
                            product_id = getattr(instr_config, 'databento_product_id', f"{symbol}.FUT")
                        else:
                            product_id = f"{symbol}.FUT"
                        request_symbols = [product_id]
                        stype_in = "parent"

                    # Try schemas in order until one works
                    logger.info(f"  {symbol}: requesting from Databento: symbols={request_symbols}, stype_in={stype_in}, dataset={dataset}")
                    data = None
                    for schema in schema_preferences:
                        try:
                            data = client.timeseries.get_range(
                                dataset=dataset,
                                schema=schema,
                                symbols=request_symbols,
                                start=start,
                                end=end,
                                stype_in=stype_in
                            )
                            if data is not None:
                                logger.info(f"  {symbol}: got data with schema {schema}")
                                break
                            else:
                                logger.info(f"  {symbol}: schema {schema} returned None")
                        except Exception as e:
                            logger.warning(f"  {symbol}: schema {schema} failed: {e}")
                            continue

                    if data is not None:
                        # Convert Databento data to Bar objects
                        try:
                            bars = self._convert_databento_to_bars(symbol, data, compression="1d")
                        except Exception as e:
                            logger.warning(f"Failed to convert daily warmup data for {symbol}: {e}")
                            del data  # Free memory even on error
                            continue

                        # Free raw Databento data immediately after conversion
                        del data

                        if not bars:
                            logger.info(f"  {symbol}: no bars after conversion (got empty list)")
                            continue

                        logger.info(f"  {symbol}: converted {len(bars)} bars, looking for feed '{symbol}_day'")

                        # Find the daily feed for this symbol
                        feed_name = f"{symbol}_day"
                        if symbol == "ZB" and "TLT_day" in self.data_feeds:
                            # Special case: ZB data goes to TLT_day feed
                            feed_name = "TLT_day"

                        if feed_name in self.data_feeds:
                            feed = self.data_feeds[feed_name]

                            # Batch the warmup loading to avoid overwhelming the queue
                            batch_size = max(int(self.warmup_batch_size), 1)
                            queue_limit = max(int(self.warmup_queue_soft_limit), batch_size)

                            total_appended = 0
                            total_batches = max((len(bars) + batch_size - 1) // batch_size, 1)

                            for batch_index, batch_start in enumerate(range(0, len(bars), batch_size)):
                                chunk = bars[batch_start : batch_start + batch_size]
                                if not chunk:
                                    continue

                                # Wait for capacity before adding more bars (except first batch)
                                if total_appended > 0:
                                    remaining_batches = max(total_batches - batch_index, 1)
                                    self._wait_for_warmup_capacity(
                                        feed,
                                        symbol=symbol,
                                        incoming=len(chunk),
                                        limit=queue_limit,
                                        pending_batches=remaining_batches,
                                        total_batches=total_batches,
                                    )

                                appended = int(feed.extend_warmup(chunk))
                                total_appended += appended

                            logger.info(f"  {symbol}: loaded {total_appended} daily warmup bars")
                        else:
                            logger.warning(f"  {symbol}: feed '{feed_name}' not found in self.data_feeds (have {list(self.data_feeds.keys())})")
                    else:
                        logger.info(f"  {symbol}: no daily data available from Databento")

                except Exception as e:
                    logger.warning(f"Failed to load daily warmup for {symbol}: {e}")
                finally:
                    # Force garbage collection after each symbol to free memory
                    gc.collect()

            # Load hourly bars (15 calendar days)
            hourly_lookback = self.strategy_config.historical_hourly_lookback_days
            end_hourly = datetime.now(tz=timezone.utc) - timedelta(hours=24)
            start_hourly = end_hourly - timedelta(days=hourly_lookback)

            logger.info(f"Loading hourly warmup data ({hourly_lookback} days)...")

            symbols_completed = 0
            for symbol in all_symbols:
                symbols_completed += 1
                logger.info(f"Loading hourly warmup for {symbol} ({symbols_completed}/{len(all_symbols)})...")
                try:
                    # Resolve symbol using contract map
                    # For OHLCV data, stype_in="parent" is safe and only returns front month
                    request_symbols: List[str] = []
                    stype_in = "parent"  # Default

                    if self.contract_map is not None:
                        subscription = self.contract_map.subscription_for(symbol)
                        if subscription and subscription.codes:
                            # Use subscription codes and stype_in from contract map
                            request_symbols.extend(subscription.codes)
                            stype_in = subscription.stype_in

                    # Fallback: use product_id from config
                    if not request_symbols:
                        instr_config = self.config.get_instrument(symbol)
                        if instr_config:
                            product_id = getattr(instr_config, 'databento_product_id', f"{symbol}.FUT")
                        else:
                            product_id = f"{symbol}.FUT"
                        request_symbols = [product_id]
                        stype_in = "parent"

                    # Try schemas in order until one works
                    logger.info(f"  {symbol}: requesting hourly from Databento: symbols={request_symbols}, stype_in={stype_in}")
                    data = None
                    for schema in schema_preferences:
                        try:
                            data = client.timeseries.get_range(
                                dataset=dataset,
                                schema=schema,
                                symbols=request_symbols,
                                start=start_hourly,
                                end=end_hourly,
                                stype_in=stype_in
                            )
                            if data is not None:
                                logger.info(f"  {symbol}: got hourly data with schema {schema}")
                                break
                            else:
                                logger.info(f"  {symbol}: schema {schema} returned None for hourly")
                        except Exception as e:
                            logger.warning(f"  {symbol}: hourly schema {schema} failed: {e}")
                            continue

                    if data is not None:
                        # Convert Databento data to Bar objects
                        try:
                            bars = self._convert_databento_to_bars(symbol, data, compression="1h")
                        except Exception as e:
                            logger.warning(f"Failed to convert hourly warmup data for {symbol}: {e}")
                            del data  # Free memory even on error
                            continue

                        # Free raw Databento data immediately after conversion
                        del data

                        if not bars:
                            logger.info(f"  {symbol}: no hourly bars after conversion (got empty list)")
                            continue

                        logger.info(f"  {symbol}: converted {len(bars)} hourly bars, looking for feed '{symbol}_hour'")

                        # Find the hourly feed for this symbol
                        feed_name = f"{symbol}_hour"
                        if feed_name in self.data_feeds:
                            feed = self.data_feeds[feed_name]

                            # Batch the warmup loading to avoid overwhelming the queue
                            batch_size = max(int(self.warmup_batch_size), 1)
                            queue_limit = max(int(self.warmup_queue_soft_limit), batch_size)

                            total_appended = 0
                            total_batches = max((len(bars) + batch_size - 1) // batch_size, 1)

                            for batch_index, batch_start in enumerate(range(0, len(bars), batch_size)):
                                chunk = bars[batch_start : batch_start + batch_size]
                                if not chunk:
                                    continue

                                # Wait for capacity before adding more bars (except first batch)
                                if total_appended > 0:
                                    remaining_batches = max(total_batches - batch_index, 1)
                                    self._wait_for_warmup_capacity(
                                        feed,
                                        symbol=symbol,
                                        incoming=len(chunk),
                                        limit=queue_limit,
                                        pending_batches=remaining_batches,
                                        total_batches=total_batches,
                                    )

                                appended = int(feed.extend_warmup(chunk))
                                total_appended += appended

                            logger.info(f"  {symbol}: loaded {total_appended} hourly warmup bars")
                        else:
                            logger.warning(f"  {symbol}: feed '{feed_name}' not found in self.data_feeds")
                    else:
                        logger.info(f"  {symbol}: no hourly data available from Databento")

                except Exception as e:
                    logger.warning(f"Failed to load hourly warmup for {symbol}: {e}")
                finally:
                    # Force garbage collection after each symbol to free memory
                    gc.collect()

            logger.info("Historical warmup complete")

            # Enable fast warmup mode - Cerebro will drain bars quickly when it starts
            self._set_strategies_warmup_mode(enabled=True)

        except ImportError:
            logger.error("databento module not found, cannot load warmup data")
        except Exception as e:
            logger.error(f"Failed to load historical warmup: {e}", exc_info=True)

    def _on_order(self, symbol: str, order):
        """Callback when order is placed/executed."""
        # Send to TradersPost
        if self.traderspost_client and order.status in [order.Completed]:
            try:
                payload = {
                    'strategy': self.strategy_name,
                    'symbol': symbol,
                    'action': 'buy' if order.isbuy() else 'sell',
                    'quantity': order.executed.size,
                    'price': order.executed.price,
                    'timestamp': datetime.utcnow().isoformat(),
                }

                self.traderspost_client.post_order(payload)
                logger.info(f"Sent order to TradersPost: {symbol} {payload['action']} {payload['quantity']}")

            except Exception as e:
                logger.error(f"Failed to send order to TradersPost: {e}")

    def _on_trade(self, symbol: str, trade):
        """Callback when trade is closed."""
        # Send to TradersPost
        if self.traderspost_client and trade.isclosed:
            try:
                payload = {
                    'strategy': self.strategy_name,
                    'symbol': symbol,
                    'pnl': trade.pnl,
                    'pnl_pct': (trade.pnl / abs(trade.value) * 100) if trade.value else 0,
                    'timestamp': datetime.utcnow().isoformat(),
                }

                self.traderspost_client.post_trade(payload)
                logger.info(f"Sent trade to TradersPost: {symbol} P&L=${trade.pnl:.2f}")

            except Exception as e:
                logger.error(f"Failed to send trade to TradersPost: {e}")

    def _on_emergency_exit(self, reason: str, context: Dict[str, Any]):
        """Callback when emergency exit triggered (e.g., daily stop loss hit)."""
        logger.critical(
            f"EMERGENCY EXIT triggered for '{self.strategy_name}': {reason} | Context: {context}"
        )

        # Could send alert notification here
        # For now, just log

    def _on_daily_summary(self):
        """Callback when trading day rolls over."""
        logger.info(f"Trading day rollover for '{self.strategy_name}'")

    def _set_strategies_warmup_mode(self, enabled: bool) -> None:
        """Enable or disable historical warmup mode on all strategies."""
        runstrats = getattr(self.cerebro, "runstrats", None)
        if not runstrats:
            return

        for strat_list in runstrats:
            if not strat_list:
                continue
            for strategy in strat_list:
                try:
                    setattr(strategy, "_in_historical_warmup", enabled)
                except Exception:
                    logger.debug("Failed to set warmup mode on strategy", exc_info=True)

        if enabled:
            logger.info("Historical warmup mode ENABLED on all strategies (fast mode)")
        else:
            logger.info("Historical warmup mode DISABLED on all strategies (full processing)")

    def _warmup_backlog_size(self, feed: Any) -> int:
        """Get the warmup backlog size for a feed."""
        if hasattr(feed, 'warmup_backlog_size'):
            try:
                return int(feed.warmup_backlog_size())
            except Exception:
                return 0
        return 0

    def _monitor_warmup_drain(self) -> None:
        """Background thread that monitors warmup backlog and disables warmup mode when drained.

        This runs after Cerebro starts and checks all feeds for warmup bars.
        Once all warmup bars are consumed, it disables strategy warmup mode.
        """
        # Give Cerebro a moment to start processing
        time.sleep(0.5)

        # Track all feeds that might have warmup bars
        tracked_feeds: list[Any] = []

        # Add all feeds from data_feeds dict
        for feed_name, feed in self.data_feeds.items():
            if feed is not None and hasattr(feed, 'warmup_backlog_size'):
                tracked_feeds.append(feed)

        if not tracked_feeds:
            logger.debug("No feeds with warmup backlog found - skipping warmup monitoring")
            return

        logger.info(f"Monitoring {len(tracked_feeds)} feeds for warmup drain...")

        # Poll until all warmup bars are drained
        max_iterations = 200  # Safety limit
        iteration = 0

        while iteration < max_iterations and not self.stop_event.is_set():
            total_backlog = 0
            for feed in tracked_feeds:
                try:
                    backlog = self._warmup_backlog_size(feed)
                    total_backlog += backlog
                except Exception:
                    pass  # Ignore errors

            if total_backlog == 0:
                # All warmup bars have been consumed!
                logger.info("Warmup backlog drained - disabling warmup mode for full processing")
                self._set_strategies_warmup_mode(enabled=False)
                return

            iteration += 1
            time.sleep(0.1)  # Check every 100ms

        # Timeout or stop requested
        if not self.stop_event.is_set():
            logger.warning(
                f"Warmup drain monitoring timed out after {max_iterations} iterations - disabling warmup mode anyway"
            )
            self._set_strategies_warmup_mode(enabled=False)

    def run(self):
        """Run the strategy worker."""
        logger.info(f"Starting strategy worker for '{self.strategy_name}'...")

        self.start_time = datetime.utcnow()
        self.running = True

        # Start heartbeat thread
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"heartbeat-{self.strategy_name}",
            daemon=True
        )
        self.heartbeat_thread.start()

        # Start warmup monitoring thread
        warmup_monitor = threading.Thread(
            target=self._monitor_warmup_drain,
            name=f"warmup-monitor-{self.strategy_name}",
            daemon=True
        )
        warmup_monitor.start()

        # Run Cerebro (blocks until stopped)
        try:
            logger.info("Running Cerebro event loop...")
            self.cerebro.run(runonce=False)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")

        except Exception as e:
            logger.error(f"Cerebro error: {e}", exc_info=True)

        finally:
            self.running = False
            self.stop_event.set()

            # Wait for heartbeat thread
            if self.heartbeat_thread:
                self.heartbeat_thread.join(timeout=2)

            logger.info(f"Strategy worker for '{self.strategy_name}' stopped")

    def _heartbeat_loop(self):
        """Background thread that writes heartbeat file periodically."""
        while not self.stop_event.is_set():
            try:
                self._write_heartbeat()
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

            # Sleep in small chunks for quick exit
            for _ in range(self.heartbeat_interval):
                if self.stop_event.is_set():
                    break
                time.sleep(1)

    def _write_heartbeat(self):
        """Write heartbeat status to file."""
        heartbeat_data = {
            'strategy': self.strategy_name,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'running' if self.running else 'stopped',
            'uptime_seconds': (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0,
            'broker_value': self.cerebro.broker.getvalue() if self.cerebro else 0,
            'broker_cash': self.cerebro.broker.getcash() if self.cerebro else 0,
        }

        # Add portfolio coordinator stats
        if self.portfolio_coordinator:
            status = self.portfolio_coordinator.get_status()
            heartbeat_data['portfolio'] = {
                'open_positions': status['open_positions_count'],
                'daily_pnl': status['daily_pnl'],
                'stopped_out': status['stopped_out'],
                'total_entries_allowed': status['stats']['total_entries_allowed'],
                'total_entries_blocked': status['stats']['total_entries_blocked'],
            }

        # Write to file
        heartbeat_path = Path(self.heartbeat_file)
        heartbeat_path.parent.mkdir(parents=True, exist_ok=True)

        with open(heartbeat_path, 'w') as f:
            json.dump(heartbeat_data, f, indent=2)

    def stop(self):
        """Stop the strategy worker gracefully."""
        logger.info(f"Stopping strategy worker for '{self.strategy_name}'...")
        self.running = False
        self.stop_event.set()

        # Stop Cerebro (if running)
        # Note: Cerebro doesn't have a native stop method for live mode
        # In practice, we'd need to signal the strategy to close positions and exit

        logger.info("Strategy worker stopped")


def main():
    """Main entry point for strategy worker."""
    parser = argparse.ArgumentParser(
        description="Strategy Worker - Run a single trading strategy"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="Name of strategy to run (e.g., 'ibs', 'breakout')"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.multi_alpha.yml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    args = parser.parse_args()

    # Create log directory if it doesn't exist
    log_dir = Path("/var/log/rooney")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=f'%(asctime)s - [{args.strategy}] - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / f"{args.strategy}_worker.log")
        ]
    )

    logger.info("=" * 80)
    logger.info(f"Starting Strategy Worker: {args.strategy}")
    logger.info("=" * 80)

    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Create worker
    try:
        worker = StrategyWorker(
            strategy_name=args.strategy,
            config=config,
            log_level=args.log_level
        )
    except Exception as e:
        logger.error(f"Failed to create strategy worker: {e}")
        sys.exit(1)

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        worker.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Setup worker
    try:
        worker.setup()
    except Exception as e:
        logger.error(f"Failed to setup strategy worker: {e}", exc_info=True)
        sys.exit(1)

    # Run worker
    logger.info(f"Strategy worker '{args.strategy}' is running. Press Ctrl+C to stop.")
    try:
        worker.run()
    except Exception as e:
        logger.error(f"Strategy worker error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
