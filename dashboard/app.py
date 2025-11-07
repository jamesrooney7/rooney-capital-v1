"""Real-time trading system dashboard built with Streamlit.

This dashboard monitors the pine-runner service and displays:
- System status and uptime
- Recent trades (entries and exits)
- Open positions
- ML filter statistics
- TradersPost webhook status
- Recent errors

Run with: streamlit run app.py
"""

from __future__ import annotations

import math
import streamlit as st
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from utils import (
    read_heartbeat,
    get_service_status,
    parse_recent_logs,
    calculate_ml_stats,
    get_recent_trades,
    get_open_positions,
)

# Import statistical monitor
from statistical_monitor import (
    StatisticalMonitor,
    get_portfolio_baseline,
    load_portfolio_baseline_detailed,
)

# Import database and metrics - use absolute path to avoid conflict with local utils.py
dashboard_path = Path(__file__).resolve().parent
src_path = dashboard_path.parent / "src"

sys.path.insert(0, str(src_path))
sys.path.insert(0, str(dashboard_path))

try:
    # Import from src/utils/ package
    import importlib.util

    # Load trades_db module
    trades_db_path = src_path / "utils" / "trades_db.py"
    spec = importlib.util.spec_from_file_location("trades_db", trades_db_path)
    trades_db_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trades_db_module)
    TradesDB = trades_db_module.TradesDB

    # Load metrics module
    metrics_path = dashboard_path / "metrics.py"
    spec = importlib.util.spec_from_file_location("dashboard_metrics", metrics_path)
    metrics_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics_module)
    calculate_portfolio_metrics = metrics_module.calculate_portfolio_metrics
    calculate_instrument_metrics = metrics_module.calculate_instrument_metrics

    DB_AVAILABLE = True
except Exception as e:
    DB_AVAILABLE = False
    # Only show error in sidebar, not main page
    st.sidebar.error(f"Database unavailable: {e}")


# Page configuration
st.set_page_config(
    page_title="Rooney Exploration Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Title
st.title("📊 Rooney Exploration Dashboard")
st.caption("Real-time portfolio monitoring and performance analytics")

# Auto-refresh every 10 seconds
st.markdown(
    """
    <script>
        setTimeout(function() {
            window.location.reload();
        }, 10000);
    </script>
    """,
    unsafe_allow_html=True,
)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    log_lines = st.slider("Log lines to fetch", 500, 5000, 1000, 500)
    log_window = st.slider("Time window (minutes)", 30, 720, 60, 30)
    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Fetch data
heartbeat = read_heartbeat()
service_status = get_service_status()
log_data = parse_recent_logs(lines=log_lines, since_minutes=log_window)
ml_stats = calculate_ml_stats(log_data)

# Initialize recent_trades and open_positions from logs as fallback
recent_trades_from_logs = get_recent_trades(log_data, limit=10)
open_positions_from_logs = get_open_positions(log_data)

# Will override with database if available
recent_trades = recent_trades_from_logs
open_positions = open_positions_from_logs

# Fetch database metrics if available
db_trades = []
portfolio_metrics = {}
instrument_metrics = {}
if DB_AVAILABLE:
    try:
        db = TradesDB()
        db_trades = db.get_all_trades()
        daily_pnl = db.get_daily_pnl()

        # Get open positions from database (trades without exit_time)
        db_open_positions = [
            {
                'symbol': trade['symbol'],
                'size': 1.0,  # Default size
                'entry_price': trade['entry_price'],
                'entry_time': datetime.fromisoformat(trade['entry_time']) if trade.get('entry_time') else None
            }
            for trade in db_trades
            if not trade.get('exit_time') and trade.get('entry_time')  # Has entry but no exit
        ]

        # Debug: Show how many open vs closed trades
        if db_trades:
            open_count = sum(1 for t in db_trades if not t.get('exit_time'))
            closed_count = sum(1 for t in db_trades if t.get('exit_time'))
            st.sidebar.info(f"Debug: {open_count} open, {closed_count} closed trades in DB")

        # Get recent trades (show both entries and exits, last 20 events)
        db_recent_trades = []
        completed_trades = [t for t in db_trades if t.get('exit_time')][:20]
        for trade in completed_trades:
            # Add entry notification
            db_recent_trades.append({
                'type': 'entry',
                'symbol': trade['symbol'],
                'action': 'buy',  # Assuming long positions
                'price': trade['entry_price'],
                'size': 1.0,
                'timestamp': datetime.fromisoformat(trade['entry_time']) if trade.get('entry_time') else None,
                'ibs': trade.get('ibs_entry'),
                'ml_score': trade.get('ml_score'),
                'reason': None
            })
            # Add exit notification
            db_recent_trades.append({
                'type': 'exit',
                'symbol': trade['symbol'],
                'action': 'sell',
                'price': trade['exit_price'],
                'size': 1.0,
                'timestamp': datetime.fromisoformat(trade['exit_time']) if trade.get('exit_time') else None,
                'ibs': trade.get('ibs_exit'),
                'ml_score': None,
                'reason': trade.get('exit_reason', 'Trade closed')
            })

        # Sort by timestamp descending (most recent first)
        db_recent_trades.sort(key=lambda x: x['timestamp'] if x['timestamp'] else datetime.min, reverse=True)

        # Override with database data if available
        if db_open_positions:
            open_positions = db_open_positions
        if db_recent_trades:
            recent_trades = db_recent_trades

        # Calculate portfolio metrics
        portfolio_metrics = calculate_portfolio_metrics(
            trades=db_trades,
            daily_pnl=daily_pnl,
            starting_capital=150000.0,  # Actual account size for accurate Sharpe calculation
            risk_free_rate=0.04,  # 4% annual risk-free rate
        )

        # Calculate per-instrument metrics
        instrument_metrics = calculate_instrument_metrics(db_trades)
    except Exception as e:
        st.sidebar.error(f"Failed to load metrics: {e}")
        DB_AVAILABLE = False


# ============================================================================
# System Status Section
# ============================================================================

st.header("🖥️ System Status")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if service_status["is_running"]:
        st.metric("Service Status", "🟢 Running", delta=None)
    else:
        st.metric("Service Status", "🔴 Down", delta=None)

with col2:
    st.metric("Uptime", service_status["uptime"])

with col3:
    heartbeat_status = heartbeat.get("status", "unknown")
    if heartbeat_status == "running":
        st.metric("Heartbeat", "🟢 Active")
    elif heartbeat_status == "unknown":
        st.metric("Heartbeat", "⚪ Unknown")
    else:
        st.metric("Heartbeat", f"⚠️ {heartbeat_status.title()}")

with col4:
    updated_at = heartbeat.get("updated_at")
    if updated_at:
        try:
            dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            seconds_ago = (datetime.now(dt.tzinfo) - dt).total_seconds()
            if seconds_ago < 60:
                st.metric("Last Heartbeat", f"{int(seconds_ago)}s ago")
            elif seconds_ago < 3600:
                st.metric("Last Heartbeat", f"{int(seconds_ago/60)}m ago")
            else:
                st.metric("Last Heartbeat", f"{int(seconds_ago/3600)}h ago")
        except (ValueError, AttributeError):
            st.metric("Last Heartbeat", "Unknown")
    else:
        st.metric("Last Heartbeat", "No data")


# ============================================================================
# Service Details
# ============================================================================

with st.expander("📡 Service Details"):
    # Both TradersPost and Databento are under "details" key
    details = heartbeat.get("details", {})
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("TradersPost Webhook")
        tp_status = details.get("traderspost", {})
        if isinstance(tp_status, dict):
            last_success = tp_status.get("last_success", {})
            last_error = tp_status.get("last_error")

            # Configured if last_success exists
            configured = last_success is not None and isinstance(last_success, dict)
            st.write(f"**Configured:** {'✅ Yes' if configured else '❌ No'}")

            # Last post timestamp
            if configured and last_success.get("at"):
                st.write(f"**Last Post:** {last_success.get('at', 'Never')}")
            else:
                st.write(f"**Last Post:** Never")

            # Show last error if exists
            if last_error:
                st.write(f"**Last Error:** {last_error}")
            else:
                st.write(f"**Status:** ✅ No errors")
        else:
            st.write("No webhook data available")

    with col2:
        st.subheader("Databento Feed")
        db_status = details.get("databento", {})
        if isinstance(db_status, dict):
            # Get subscriber info (more detailed than queue_fanout)
            subscribers = db_status.get("subscribers", [])
            queue_fanout = db_status.get("queue_fanout", {})
            known_symbols = queue_fanout.get("known_symbols", [])

            # Check connection from subscribers
            client_connected = False
            last_trade_time = None
            if subscribers and len(subscribers) > 0:
                subscriber = subscribers[0]
                client_connected = subscriber.get("client_connected", False)
                last_trade = subscriber.get("last_trade", {})
                if last_trade:
                    # Get most recent trade time across all symbols
                    times = [t for t in last_trade.values() if t]
                    if times:
                        last_trade_time = max(times)

            st.write(f"**Connected:** {'✅ Yes' if client_connected else '❌ No'}")
            st.write(f"**Symbols:** {len(known_symbols)} tracked")
            if last_trade_time:
                st.write(f"**Last Trade:** {last_trade_time}")

            if known_symbols and len(known_symbols) <= 12:
                st.write(f"**List:** {', '.join(known_symbols)}")
            elif known_symbols:
                st.write(f"**List:** {', '.join(known_symbols[:12])}, ...")
        else:
            st.write("No data feed info available")


# ============================================================================
# Open Positions
# ============================================================================

st.header("📈 Open Positions")

if open_positions:
    for pos in open_positions:
        col1, col2, col3, col4 = st.columns([2, 2, 2, 4])

        with col1:
            st.write(f"**{pos['symbol']}**")

        with col2:
            st.write(f"Size: {pos['size']:.0f}")

        with col3:
            if pos['entry_price']:
                st.write(f"Entry: ${pos['entry_price']:.2f}")
            else:
                st.write("Entry: Unknown")

        with col4:
            if pos['entry_time']:
                st.write(f"Opened: {pos['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.write("Opened: Unknown")

        st.divider()
else:
    st.info("No open positions")


# ============================================================================
# Recent Trades
# ============================================================================

st.header("📋 Recent Trades")

if recent_trades:
    for trade in recent_trades:
        # Color code by trade type
        if trade["type"] == "entry":
            color = "🟢"
            type_label = "ENTRY"
        else:
            color = "🔴"
            type_label = "EXIT"

        # Create expandable trade row
        timestamp_str = trade["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if trade["timestamp"] else "Unknown"

        with st.expander(
            f"{color} {trade['symbol']} {type_label} @ ${trade['price']:.2f} | {timestamp_str}"
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Symbol:** {trade['symbol']}")
                st.write(f"**Action:** {trade['action'].upper()}")
                st.write(f"**Price:** ${trade['price']:.2f}")
                st.write(f"**Size:** {trade['size']:.0f}")

            with col2:
                st.write(f"**Timestamp:** {timestamp_str}")

                if trade["type"] == "entry":
                    if trade.get("ibs") is not None:
                        st.write(f"**IBS Value:** {trade['ibs']:.3f}")
                    if trade.get("ml_score") is not None:
                        st.write(f"**ML Score:** {trade['ml_score']:.3f}")
                else:
                    if trade.get("ibs") is not None:
                        st.write(f"**Exit IBS:** {trade['ibs']:.3f}")
                    if trade.get("reason"):
                        st.write(f"**Exit Reason:** {trade['reason']}")
else:
    st.info(f"No trades in the last {log_window} minutes")


# ============================================================================
# Blocked Trades
# ============================================================================

if log_data["ml_blocked"]:
    st.header("⛔ ML Filter Blocked Trades")

    for blocked in log_data["ml_blocked"][:5]:  # Show last 5 blocked
        timestamp_str = blocked["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if blocked["timestamp"] else "Unknown"

        st.warning(
            f"**{blocked['symbol']}** blocked at {timestamp_str} | "
            f"Score: {blocked['ml_score']:.3f} < Threshold: {blocked['threshold']:.3f}"
        )


# ============================================================================
# Recent Errors
# ============================================================================

if log_data["errors"]:
    st.header("⚠️ Recent Errors")

    with st.expander(f"Show {len(log_data['errors'])} errors"):
        for error in log_data["errors"][:10]:  # Show last 10 errors
            timestamp_str = error["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if error["timestamp"] else "Unknown"
            st.error(f"[{timestamp_str}] {error['raw']}")


# ============================================================================
# Portfolio Performance
# ============================================================================

if DB_AVAILABLE and portfolio_metrics:
    st.header("📈 Portfolio Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total P&L",
            f"${portfolio_metrics.get('total_pnl', 0):.2f}",
            help="Cumulative realized profit/loss"
        )

    with col2:
        sharpe = portfolio_metrics.get('sharpe_ratio', 0)
        sharpe_color = "🟢" if sharpe > 1.0 else "🟡" if sharpe > 0.5 else "🔴"
        st.metric(
            "Sharpe Ratio",
            f"{sharpe_color} {sharpe:.2f}",
            help="Risk-adjusted returns (>1.0 is good)"
        )

    with col3:
        sortino = portfolio_metrics.get('sortino_ratio', 0)
        sortino_color = "🟢" if sortino > 1.5 else "🟡" if sortino > 0.75 else "🔴"
        st.metric(
            "Sortino Ratio",
            f"{sortino_color} {sortino:.2f}",
            help="Downside risk-adjusted returns (>1.5 is good)"
        )

    with col4:
        pf = portfolio_metrics.get('profit_factor', 0)
        pf_color = "🟢" if pf > 1.5 else "🟡" if pf > 1.0 else "🔴"
        pf_display = f"{pf:.2f}" if pf != float('inf') else "∞"
        st.metric(
            "Profit Factor",
            f"{pf_color} {pf_display}",
            help="Gross profit / Gross loss (>1.5 is good)"
        )

    # Second row of metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Win Rate",
            f"{portfolio_metrics.get('win_rate', 0):.1f}%",
            help="Percentage of profitable trades"
        )

    with col2:
        st.metric(
            "Total Trades",
            f"{portfolio_metrics.get('total_trades', 0)}",
            help="Number of completed round-trip trades"
        )

    with col3:
        max_dd = portfolio_metrics.get('max_drawdown', 0)
        max_dd_pct = portfolio_metrics.get('max_drawdown_pct', 0)
        st.metric(
            "Max Drawdown",
            f"${max_dd:.2f}",
            delta=f"-{max_dd_pct:.1f}%",
            delta_color="inverse",
            help="Maximum peak-to-trough decline"
        )

    with col4:
        avg_duration = portfolio_metrics.get('avg_trade_duration_hours', 0)
        st.metric(
            "Avg Duration",
            f"{avg_duration:.1f}h",
            help="Average time in trade (hours)"
        )

    # Additional comprehensive stats in expandable sections
    with st.expander("📊 Risk & Return Metrics"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            expectancy = portfolio_metrics.get('expectancy', 0)
            exp_color = "🟢" if expectancy > 0 else "🔴"
            st.write(f"**Expectancy:** {exp_color} ${expectancy:.2f}")

        with col2:
            calmar = portfolio_metrics.get('calmar_ratio', 0)
            st.write(f"**Calmar Ratio:** {calmar:.2f}")

        with col3:
            recovery = portfolio_metrics.get('recovery_factor', 0)
            rec_display = f"{recovery:.2f}" if recovery != float('inf') else "∞"
            st.write(f"**Recovery Factor:** {rec_display}")

        with col4:
            st.write(f"**Trades/Day:** {portfolio_metrics.get('trades_per_day', 0):.2f}")

    with st.expander("💰 Win/Loss Analysis"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.write(f"**Avg Win:** ${portfolio_metrics.get('avg_win', 0):.2f}")
        with col2:
            st.write(f"**Avg Loss:** ${portfolio_metrics.get('avg_loss', 0):.2f}")
        with col3:
            wl_ratio = portfolio_metrics.get('win_loss_ratio', 0)
            st.write(f"**Win/Loss Ratio:** {wl_ratio:.2f}x")
        with col4:
            best_trade = portfolio_metrics.get('best_trade', 0)
            st.write(f"**Best Trade:** ${best_trade:.2f}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            worst_trade = portfolio_metrics.get('worst_trade', 0)
            st.write(f"**Worst Trade:** ${worst_trade:.2f}")
        with col2:
            best_day = portfolio_metrics.get('best_day', 0)
            st.write(f"**Best Day:** ${best_day:.2f}")
        with col3:
            worst_day = portfolio_metrics.get('worst_day', 0)
            st.write(f"**Worst Day:** ${worst_day:.2f}")
        with col4:
            profitable_days_pct = portfolio_metrics.get('profitable_days_pct', 0)
            st.write(f"**Profitable Days:** {profitable_days_pct:.1f}%")

    with st.expander("🔥 Streaks"):
        col1, col2, col3 = st.columns(3)

        with col1:
            current_streak = portfolio_metrics.get('current_streak', 0)
            if current_streak > 0:
                st.write(f"**Current Streak:** 🟢 {current_streak} wins")
            elif current_streak < 0:
                st.write(f"**Current Streak:** 🔴 {abs(current_streak)} losses")
            else:
                st.write(f"**Current Streak:** ⚪ 0")

        with col2:
            max_win = portfolio_metrics.get('max_win_streak', 0)
            st.write(f"**Max Win Streak:** {max_win}")

        with col3:
            max_loss = portfolio_metrics.get('max_loss_streak', 0)
            st.write(f"**Max Loss Streak:** {max_loss}")


# ============================================================================
# Statistical Quality Control
# ============================================================================

# Debug information
if not DB_AVAILABLE:
    st.header("📊 Statistical Quality Control")
    st.error("❌ Database not available. Check sidebar for error message.")
elif not db_trades:
    st.header("📊 Statistical Quality Control")
    st.info("ℹ️ No trades in database yet. Statistical monitoring will activate once trades are recorded.")
elif DB_AVAILABLE and db_trades:
    st.header("📊 Statistical Quality Control")
    st.caption("Compare live performance to backtest expectations with statistical rigor")

    try:
        # Initialize monitor (results dir is in parent directory)
        monitor = StatisticalMonitor(results_dir="../results")

        # Load detailed portfolio baseline
        portfolio_baseline_detailed = load_portfolio_baseline_detailed(results_dir="../results")

        if not portfolio_baseline_detailed:
            st.warning("⚠️ Could not load portfolio baseline from results directory. Check that results/ directory exists with greedy_optimization_*.json file.")
    except Exception as e:
        st.error(f"❌ Error initializing statistical monitor: {e}")
        import traceback
        st.code(traceback.format_exc())
        portfolio_baseline_detailed = None

    # --- PORTFOLIO-LEVEL MONITORING (PRIMARY) ---
    if portfolio_baseline_detailed and portfolio_metrics and portfolio_metrics.get('total_trades', 0) >= 5:
        st.subheader("📈 Portfolio-Level Quality Control")

        # Create pseudo-objects for portfolio comparison
        from statistical_monitor import BacktestBaseline, LivePerformance

        # Convert portfolio baseline to BacktestBaseline format
        portfolio_bl = BacktestBaseline(
            symbol="PORTFOLIO",
            sharpe=portfolio_baseline_detailed.sharpe,
            profit_factor=0.0,  # Not available in portfolio baseline
            trades=portfolio_baseline_detailed.trades,
            total_pnl=portfolio_baseline_detailed.total_pnl,
            max_drawdown=portfolio_baseline_detailed.max_drawdown,
            win_rate=portfolio_baseline_detailed.win_rate,
            avg_return=portfolio_baseline_detailed.avg_return,
            std_return=portfolio_baseline_detailed.std_return,
            threshold=0.65,
        )

        # Get all trade PnLs for returns calculation (sorted by exit time, newest first)
        completed_trades_sorted = sorted(
            [t for t in db_trades if t.get("pnl") != 0 and t.get("exit_time")],
            key=lambda x: x.get("exit_time", ""),
            reverse=True
        )
        all_trade_pnls = [t["pnl"] for t in completed_trades_sorted]

        # Calculate rolling 50 trades if we have enough data
        rolling_50_pnls = all_trade_pnls[:50] if len(all_trade_pnls) >= 50 else None

        # Convert portfolio metrics to LivePerformance format (ALL-TIME)
        portfolio_live = LivePerformance(
            symbol="PORTFOLIO",
            n_trades=portfolio_metrics.get('total_trades', 0),
            sharpe=portfolio_metrics.get('sharpe_ratio', 0.0),
            profit_factor=portfolio_metrics.get('profit_factor', 0.0),
            win_rate=portfolio_metrics.get('win_rate', 0.0),
            total_pnl=portfolio_metrics.get('total_pnl', 0.0),
            max_drawdown=portfolio_metrics.get('max_drawdown', 0.0),
            avg_return=portfolio_metrics.get('expectancy', 0.0),
            std_return=float(np.std(all_trade_pnls)) if all_trade_pnls else 0.0,
            returns=all_trade_pnls,
        )

        # Calculate rolling 50 LivePerformance if available
        portfolio_live_rolling = None
        if rolling_50_pnls:
            rolling_array = np.array(rolling_50_pnls)
            rolling_win_rate = float((rolling_array > 0).sum() / len(rolling_50_pnls) * 100)
            rolling_avg_return = float(rolling_array.mean())
            rolling_std_return = float(rolling_array.std(ddof=1))
            rolling_sharpe = rolling_avg_return / rolling_std_return * math.sqrt(252) if rolling_std_return > 0 else 0.0

            gross_profit = float(rolling_array[rolling_array > 0].sum())
            gross_loss = float(abs(rolling_array[rolling_array < 0].sum()))
            rolling_profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            cumsum = rolling_array.cumsum()
            running_max = np.maximum.accumulate(cumsum)
            drawdown = running_max - cumsum
            rolling_max_drawdown = float(drawdown.max())

            portfolio_live_rolling = LivePerformance(
                symbol="PORTFOLIO_ROLLING_50",
                n_trades=len(rolling_50_pnls),
                sharpe=rolling_sharpe,
                profit_factor=rolling_profit_factor,
                win_rate=rolling_win_rate,
                total_pnl=float(rolling_array.sum()),
                max_drawdown=rolling_max_drawdown,
                avg_return=rolling_avg_return,
                std_return=rolling_std_return,
                returns=rolling_50_pnls,
            )

        # Run statistical tests for ALL-TIME
        portfolio_tests = monitor.run_statistical_tests(portfolio_bl, portfolio_live, confidence=0.95)

        # Run statistical tests for ROLLING 50 if available
        portfolio_tests_rolling = None
        portfolio_status_rolling = None
        if portfolio_live_rolling:
            portfolio_tests_rolling = monitor.run_statistical_tests(portfolio_bl, portfolio_live_rolling, confidence=0.95)

            # Determine rolling status
            portfolio_status_rolling = "green"
            for test in portfolio_tests_rolling:
                if test.status == "red":
                    portfolio_status_rolling = "red"
                    break
                elif test.status == "yellow" and portfolio_status_rolling != "red":
                    portfolio_status_rolling = "yellow"

        # Determine overall portfolio status (all-time)
        portfolio_status = "green"
        for test in portfolio_tests:
            if test.status == "red":
                portfolio_status = "red"
                break
            elif test.status == "yellow" and portfolio_status != "red":
                portfolio_status = "yellow"

        # Display status banners
        col_banner1, col_banner2 = st.columns(2)

        with col_banner1:
            st.markdown("**All-Time Performance**")
            if portfolio_status == "green":
                st.success("🟢 HEALTHY - Matches backtest")
            elif portfolio_status == "yellow":
                st.warning("🟡 CAUTION - Some deviation")
            else:
                st.error("🔴 ALERT - Significant deviation")

        with col_banner2:
            if portfolio_live_rolling:
                st.markdown("**Recent Performance (Last 50 Trades)**")
                if portfolio_status_rolling == "green":
                    st.success("🟢 HEALTHY - Matches backtest")
                elif portfolio_status_rolling == "yellow":
                    st.warning("🟡 CAUTION - Some deviation")
                else:
                    st.error("🔴 ALERT - Significant deviation")
            else:
                st.markdown("**Recent Performance (Last 50 Trades)**")
                st.info(f"⏳ Need 50 trades (have {len(all_trade_pnls)})")

        # Show key metrics comparison - SIDE BY SIDE
        st.markdown("---")

        # Create tabs for All-Time vs Rolling 50
        if portfolio_live_rolling:
            tab1, tab2 = st.tabs(["📊 All-Time", "🔄 Rolling 50 Trades"])
        else:
            tab1, tab2 = st.tabs(["📊 All-Time", "🔄 Rolling 50 Trades (Not Ready)"])

        with tab1:
            st.caption("Cumulative performance since going live")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Live Trades",
                    f"{portfolio_live.n_trades}",
                    help=f"Baseline: {portfolio_bl.trades} trades in test period"
                )

            with col2:
                sharpe_delta = portfolio_live.sharpe - portfolio_bl.sharpe
                st.metric(
                    "Sharpe Ratio",
                    f"{portfolio_live.sharpe:.2f}",
                    delta=f"{sharpe_delta:+.2f}",
                    help=f"Expected: {portfolio_bl.sharpe:.2f}"
                )

            with col3:
                wr_delta = portfolio_live.win_rate - portfolio_bl.win_rate
                st.metric(
                    "Win Rate",
                    f"{portfolio_live.win_rate:.1f}%",
                    delta=f"{wr_delta:+.1f}%",
                    help=f"Expected: {portfolio_bl.win_rate:.1f}%"
                )

            with col4:
                # Sample size recommendation
                rec_n = monitor.get_sample_size_recommendation(portfolio_bl, power=0.80, alpha=0.05)
                if portfolio_live.n_trades >= rec_n:
                    st.metric("Sample Size", "✅ Sufficient", help=f"Have {portfolio_live.n_trades} ≥ {rec_n} needed")
                else:
                    st.metric("Sample Size", f"⚠️ {portfolio_live.n_trades}/{rec_n}", help="Need more trades for full power")

            # Detailed test results in expander
            with st.expander("📊 Detailed Statistical Tests (All-Time)"):
                for test in portfolio_tests:
                    # Color code
                    if test.status == "green":
                        color = "#d4edda"
                        border = "#c3e6cb"
                    elif test.status == "yellow":
                        color = "#fff3cd"
                        border = "#ffeaa7"
                    else:
                        color = "#f8d7da"
                        border = "#f5c6cb"

                    # Format values
                    if test.metric_name == "Average Return":
                        obs_str = f"${test.observed:.2f}"
                        exp_str = f"${test.expected:.2f}"
                        ci_str = f"[${test.ci_lower:.2f}, ${test.ci_upper:.2f}]"
                    else:
                        obs_str = f"{test.observed:.2f}"
                        exp_str = f"{test.expected:.2f}"
                        ci_str = f"[{test.ci_lower:.2f}, {test.ci_upper:.2f}]"

                    st.markdown(
                        f"""
                        <div style='background-color: {color}; padding: 10px; border-radius: 5px;
                                    border: 1px solid {border}; margin-bottom: 10px;'>
                            <b>{test.metric_name}</b><br>
                            Observed: {obs_str} | Expected: {exp_str}<br>
                            Z-score: {test.z_score:.2f} | P-value: {test.p_value:.4f}<br>
                            95% CI: {ci_str}<br>
                            <b>Status: {test.status.upper()}</b>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # SPRT for portfolio (all-time)
            with st.expander("🔬 Sequential Testing (SPRT - All-Time)"):
                sprt_result = monitor.run_sprt(portfolio_bl, portfolio_live, alpha=0.05, beta=0.20)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write(f"**Log-Likelihood Ratio:** {sprt_result['log_likelihood_ratio']:.2f}")

                with col2:
                    st.write(f"**Upper Threshold (H0):** {sprt_result['threshold_upper']:.2f}")
                    st.write(f"**Lower Threshold (H1):** {sprt_result['threshold_lower']:.2f}")

                with col3:
                    decision = sprt_result['decision']
                    if decision == "accept_H0":
                        st.success("✅ PASS - Performance OK")
                    elif decision == "accept_H1":
                        st.error("❌ FAIL - Performance degraded")
                    else:
                        st.info("⏳ CONTINUE - Keep monitoring")

                st.caption(sprt_result['message'])

        # TAB 2: Rolling 50 trades
        with tab2:
            if portfolio_live_rolling:
                st.caption("Most recent 50 completed trades")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Trades",
                        "50",
                        help="Rolling window of last 50 trades"
                    )

                with col2:
                    sharpe_delta_rolling = portfolio_live_rolling.sharpe - portfolio_bl.sharpe
                    st.metric(
                        "Sharpe Ratio",
                        f"{portfolio_live_rolling.sharpe:.2f}",
                        delta=f"{sharpe_delta_rolling:+.2f}",
                        help=f"Expected: {portfolio_bl.sharpe:.2f}"
                    )

                with col3:
                    wr_delta_rolling = portfolio_live_rolling.win_rate - portfolio_bl.win_rate
                    st.metric(
                        "Win Rate",
                        f"{portfolio_live_rolling.win_rate:.1f}%",
                        delta=f"{wr_delta_rolling:+.1f}%",
                        help=f"Expected: {portfolio_bl.win_rate:.1f}%"
                    )

                with col4:
                    st.metric("Sample Size", "✅ Sufficient", help="50 trades meets power requirements")

                # Detailed test results for rolling
                with st.expander("📊 Detailed Statistical Tests (Rolling 50)"):
                    for test in portfolio_tests_rolling:
                        # Color code
                        if test.status == "green":
                            color = "#d4edda"
                            border = "#c3e6cb"
                        elif test.status == "yellow":
                            color = "#fff3cd"
                            border = "#ffeaa7"
                        else:
                            color = "#f8d7da"
                            border = "#f5c6cb"

                        # Format values
                        if test.metric_name == "Average Return":
                            obs_str = f"${test.observed:.2f}"
                            exp_str = f"${test.expected:.2f}"
                            ci_str = f"[${test.ci_lower:.2f}, ${test.ci_upper:.2f}]"
                        else:
                            obs_str = f"{test.observed:.2f}"
                            exp_str = f"{test.expected:.2f}"
                            ci_str = f"[{test.ci_lower:.2f}, {test.ci_upper:.2f}]"

                        st.markdown(
                            f"""
                            <div style='background-color: {color}; padding: 10px; border-radius: 5px;
                                        border: 1px solid {border}; margin-bottom: 10px;'>
                                <b>{test.metric_name}</b><br>
                                Observed: {obs_str} | Expected: {exp_str}<br>
                                Z-score: {test.z_score:.2f} | P-value: {test.p_value:.4f}<br>
                                95% CI: {ci_str}<br>
                                <b>Status: {test.status.upper()}</b>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                # SPRT for rolling 50
                with st.expander("🔬 Sequential Testing (SPRT - Rolling 50)"):
                    sprt_result_rolling = monitor.run_sprt(portfolio_bl, portfolio_live_rolling, alpha=0.05, beta=0.20)

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.write(f"**Log-Likelihood Ratio:** {sprt_result_rolling['log_likelihood_ratio']:.2f}")

                    with col2:
                        st.write(f"**Upper Threshold (H0):** {sprt_result_rolling['threshold_upper']:.2f}")
                        st.write(f"**Lower Threshold (H1):** {sprt_result_rolling['threshold_lower']:.2f}")

                    with col3:
                        decision = sprt_result_rolling['decision']
                        if decision == "accept_H0":
                            st.success("✅ PASS - Performance OK")
                        elif decision == "accept_H1":
                            st.error("❌ FAIL - Performance degraded")
                        else:
                            st.info("⏳ CONTINUE - Keep monitoring")

                    st.caption(sprt_result_rolling['message'])
            else:
                st.info(f"Need 50 completed trades to activate rolling window analysis. Currently have {len(all_trade_pnls)} trades.")

        st.divider()

    elif DB_AVAILABLE and portfolio_metrics and portfolio_metrics.get('total_trades', 0) < 5:
        st.info(
            f"**Portfolio-level monitoring:** Need at least 5 completed trades "
            f"(currently {portfolio_metrics.get('total_trades', 0)}) for statistical tests"
        )
        st.divider()

    # Load basic portfolio baseline for instrument list
    portfolio_baseline = get_portfolio_baseline(results_dir="../results")

    # --- PER-INSTRUMENT MONITORING (SECONDARY, FOR DRILL-DOWN) ---
    with st.expander("🎯 Per-Instrument Quality Control (Drill-Down)", expanded=False):
        st.caption("Detailed statistical monitoring for each instrument (slower convergence)")

        # Group trades by symbol
        trades_by_symbol = {}
        for trade in db_trades:
            if not trade.get('exit_time'):
                continue  # Skip open positions
            symbol = trade.get('symbol', 'UNKNOWN')
            if symbol not in trades_by_symbol:
                trades_by_symbol[symbol] = []
            trades_by_symbol[symbol].append(trade)

        # Get instruments from portfolio baseline
        instruments = portfolio_baseline.get('instruments', [])

        if not instruments:
            # Fallback to all symbols with trades
            instruments = list(trades_by_symbol.keys())

        # Summary row
        status_counts = {"green": 0, "yellow": 0, "red": 0}

        for symbol in sorted(instruments):
            symbol_trades = trades_by_symbol.get(symbol, [])

            # Load baseline
            baseline = monitor.load_backtest_baseline(symbol)

            if not baseline:
                st.info(f"**{symbol}**: No baseline data found")
                continue

            # Calculate live performance
            live = monitor.calculate_live_performance(symbol, symbol_trades)

            if not live or live.n_trades < 5:
                st.info(
                    f"**{symbol}**: Insufficient live data "
                    f"({live.n_trades if live else 0} trades, need ≥5 for statistical tests)"
                )
                continue

            # Run statistical tests
            tests = monitor.run_statistical_tests(baseline, live, confidence=0.95)

            # Determine overall status (worst of all metrics)
            overall_status = "green"
            for test in tests:
                if test.status == "red":
                    overall_status = "red"
                    break
                elif test.status == "yellow" and overall_status != "red":
                    overall_status = "yellow"

            status_counts[overall_status] += 1

            # Status emoji
            if overall_status == "green":
                status_emoji = "🟢"
            elif overall_status == "yellow":
                status_emoji = "🟡"
            else:
                status_emoji = "🔴"

            # Expandable section per instrument
            with st.expander(
                f"{status_emoji} {symbol} | {live.n_trades} trades | Status: {overall_status.upper()}"
            ):
                # Sample size recommendation
                rec_n = monitor.get_sample_size_recommendation(baseline, power=0.80, alpha=0.05)

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.write(f"**Live Trades:** {live.n_trades}")
                    st.write(f"**Baseline Trades (Holdout):** {baseline.trades}")

                with col2:
                    if live.n_trades < rec_n:
                        st.warning(f"⚠️ Need {rec_n} trades for 80% power")
                    else:
                        st.success(f"✅ Sufficient data ({rec_n} recommended)")

                st.divider()

                # --- Phase 1: Control Charts ---
                st.write("**Phase 1: Control Charts (±2σ bands)**")

                for test in tests:
                    # Color code by status
                    if test.status == "green":
                        color = "#d4edda"  # Light green
                        border = "#c3e6cb"
                    elif test.status == "yellow":
                        color = "#fff3cd"  # Light yellow
                        border = "#ffeaa7"
                    else:
                        color = "#f8d7da"  # Light red
                        border = "#f5c6cb"

                    # Format values
                    if test.metric_name == "Average Return":
                        obs_str = f"${test.observed:.4f}"
                        exp_str = f"${test.expected:.4f}"
                        ci_str = f"[${test.ci_lower:.4f}, ${test.ci_upper:.4f}]"
                    else:
                        obs_str = f"{test.observed:.2f}"
                        exp_str = f"{test.expected:.2f}"
                        ci_str = f"[{test.ci_lower:.2f}, {test.ci_upper:.2f}]"

                    # Display metric card
                    st.markdown(
                        f"""
                        <div style='background-color: {color}; padding: 10px; border-radius: 5px;
                                    border: 1px solid {border}; margin-bottom: 10px;'>
                            <b>{test.metric_name}</b><br>
                            Observed: {obs_str} | Expected: {exp_str}<br>
                            Z-score: {test.z_score:.2f} | P-value: {test.p_value:.4f}<br>
                            95% CI: {ci_str}<br>
                            <b>Status: {test.status.upper()}</b>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.divider()

                # --- Phase 3: SPRT ---
                st.write("**Phase 3: Sequential Probability Ratio Test (SPRT)**")

                sprt_result = monitor.run_sprt(baseline, live, alpha=0.05, beta=0.20)

                # Visualize SPRT
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write(f"**LLR:** {sprt_result['log_likelihood_ratio']:.2f}")

                with col2:
                    st.write(f"**Upper Threshold:** {sprt_result['threshold_upper']:.2f}")
                    st.write(f"**Lower Threshold:** {sprt_result['threshold_lower']:.2f}")

                with col3:
                    decision = sprt_result['decision']
                    if decision == "accept_H0":
                        st.success("✅ PASS")
                    elif decision == "accept_H1":
                        st.error("❌ FAIL")
                    else:
                        st.info("⏳ CONTINUE")

                st.write(sprt_result['message'])

        # Summary box
        st.divider()
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Instruments Monitored", len(instruments))

        with col2:
            st.metric("🟢 Green", status_counts.get("green", 0))

        with col3:
            st.metric("🟡 Yellow", status_counts.get("yellow", 0))

        with col4:
            st.metric("🔴 Red", status_counts.get("red", 0))


# ============================================================================
# P&L Calendar
# ============================================================================

if DB_AVAILABLE and db_trades:
    st.header("📅 P&L Calendar")

    # Get all dates with trades
    db = TradesDB()
    all_daily_pnl = db.get_daily_pnl()

    if all_daily_pnl:
        # Month selector
        from datetime import datetime as dt_class
        dates = [dt_class.fromisoformat(d) for d in all_daily_pnl.keys()]
        min_date = min(dates)
        max_date = max(dates)

        # Create month options
        months = []
        current = min_date.replace(day=1)
        while current <= max_date:
            months.append(current)
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        # Month selector
        selected_month = st.selectbox(
            "Select Month",
            months,
            index=len(months) - 1,  # Default to most recent month
            format_func=lambda x: x.strftime("%B %Y")
        )

        # Filter P&L for selected month
        month_start = selected_month.replace(day=1)
        if selected_month.month == 12:
            month_end = selected_month.replace(year=selected_month.year + 1, month=1, day=1)
        else:
            month_end = selected_month.replace(month=selected_month.month + 1, day=1)

        # Create calendar grid
        import calendar
        cal = calendar.monthcalendar(selected_month.year, selected_month.month)

        # Build calendar
        st.write(f"### {selected_month.strftime('%B %Y')}")

        # Days of week header
        col_headers = st.columns(7)
        for i, day in enumerate(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']):
            col_headers[i].write(f"**{day}**")

        # Calendar rows
        for week in cal:
            cols = st.columns(7)
            for i, day in enumerate(week):
                if day == 0:
                    cols[i].write("")
                else:
                    date_str = f"{selected_month.year}-{selected_month.month:02d}-{day:02d}"
                    pnl = all_daily_pnl.get(date_str, 0.0)

                    if pnl > 0:
                        cols[i].markdown(
                            f"<div style='background-color: #d4edda; padding: 5px; border-radius: 3px; text-align: center;'>"
                            f"<small>{day}</small><br><b>${pnl:.0f}</b></div>",
                            unsafe_allow_html=True
                        )
                    elif pnl < 0:
                        cols[i].markdown(
                            f"<div style='background-color: #f8d7da; padding: 5px; border-radius: 3px; text-align: center;'>"
                            f"<small>{day}</small><br><b>${pnl:.0f}</b></div>",
                            unsafe_allow_html=True
                        )
                    else:
                        cols[i].markdown(
                            f"<div style='padding: 5px; text-align: center;'>"
                            f"<small style='color: #999;'>{day}</small></div>",
                            unsafe_allow_html=True
                        )

        # Month summary
        month_pnl = sum(
            pnl for date_str, pnl in all_daily_pnl.items()
            if month_start <= dt_class.fromisoformat(date_str) < month_end
        )
        month_days = sum(
            1 for date_str in all_daily_pnl.keys()
            if month_start <= dt_class.fromisoformat(date_str) < month_end
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Month P&L", f"${month_pnl:.2f}")
        with col2:
            st.metric("Trading Days", month_days)
        with col3:
            avg_day = month_pnl / month_days if month_days > 0 else 0
            st.metric("Avg Day", f"${avg_day:.2f}")


# ============================================================================
# Per-Instrument Performance
# ============================================================================

# Debug information
if not DB_AVAILABLE:
    st.header("🎯 Per-Instrument Performance")
    st.error("❌ Database not available. Check sidebar for error message.")
elif not instrument_metrics:
    st.header("🎯 Per-Instrument Performance")
    st.info("ℹ️ No completed trades yet. Per-instrument metrics will show after first trade closes.")
elif DB_AVAILABLE and instrument_metrics:
    st.header("🎯 Per-Instrument Performance")

    # Sort instruments by total P&L
    sorted_instruments = sorted(
        instrument_metrics.items(),
        key=lambda x: x[1]['total_pnl'],
        reverse=True
    )

    for symbol, metrics in sorted_instruments:
        # Color code by P&L
        pnl = metrics['total_pnl']
        pnl_color = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"

        with st.expander(f"{pnl_color} {symbol} | P&L: ${pnl:.2f} | Trades: {metrics['total_trades']}"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.write(f"**Total P&L:** ${metrics['total_pnl']:.2f}")
                st.write(f"**Win Rate:** {metrics['win_rate']:.1f}%")

            with col2:
                pf = metrics['profit_factor']
                pf_display = f"{pf:.2f}" if pf != float('inf') else "∞"
                st.write(f"**Profit Factor:** {pf_display}")
                st.write(f"**Trades:** {metrics['total_trades']}")

            with col3:
                st.write(f"**Best Trade:** ${metrics['best_trade']:.2f}")
                st.write(f"**Worst Trade:** ${metrics['worst_trade']:.2f}")

            with col4:
                avg_duration = metrics['avg_trade_duration_hours']
                st.write(f"**Avg Duration:** {avg_duration:.1f}h")


# ============================================================================
# System Health
# ============================================================================

st.header("🏥 System Health")

col1, col2, col3, col4 = st.columns(4)

with col1:
    # Trades today
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if DB_AVAILABLE:
        try:
            db = TradesDB()
            today_trades = db.get_trades_since(today_start)
            st.metric("Trades Today", len(today_trades))
        except:
            st.metric("Trades Today", "N/A")
    else:
        st.metric("Trades Today", "N/A")

with col2:
    # Error count from logs
    error_count = len(log_data.get("errors", []))
    error_color = "🟢" if error_count == 0 else "🟡" if error_count < 5 else "🔴"
    st.metric("Recent Errors", f"{error_color} {error_count}")

with col3:
    # Time since last trade
    if db_trades:
        last_trade = db_trades[0]  # Already sorted by exit_time descending
        last_exit = datetime.fromisoformat(last_trade['exit_time'])
        time_since = datetime.now() - last_exit.replace(tzinfo=None)
        hours = int(time_since.total_seconds() / 3600)
        minutes = int((time_since.total_seconds() % 3600) / 60)
        st.metric("Last Trade", f"{hours}h {minutes}m ago")
    else:
        st.metric("Last Trade", "No trades yet")

with col4:
    # ML filter effectiveness
    if ml_stats['total_signals'] > 0:
        pass_rate = ml_stats['pass_rate']
        filter_health = "🟢" if 20 <= pass_rate <= 40 else "🟡" if 10 <= pass_rate <= 50 else "🔴"
        st.metric("ML Filter Health", f"{filter_health} {pass_rate:.0f}% pass")
    else:
        st.metric("ML Filter Health", "No signals")


# ============================================================================
# Footer
# ============================================================================

st.divider()
st.caption(
    "Pine Trading System Dashboard | "
    f"Monitoring last {log_window} minutes | "
    f"Auto-refresh: 10s | "
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
