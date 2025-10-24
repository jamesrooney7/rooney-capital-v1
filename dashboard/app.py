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

import streamlit as st
from datetime import datetime

from utils import (
    read_heartbeat,
    get_service_status,
    parse_recent_logs,
    calculate_ml_stats,
    get_recent_trades,
    get_open_positions,
)


# Page configuration
st.set_page_config(
    page_title="Pine Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Title
st.title("📊 Pine Trading System Dashboard")
st.caption("Real-time monitoring for algorithmic trading")

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
recent_trades = get_recent_trades(log_data, limit=10)
open_positions = get_open_positions(log_data)


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
# ML Filter Statistics
# ============================================================================

st.header("🤖 ML Filter Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Signals",
        ml_stats["total_signals"],
        help="Total number of IBS signals in time window"
    )

with col2:
    st.metric(
        "Passed",
        ml_stats["passed"],
        delta=f"{ml_stats['pass_rate']:.1f}%",
        delta_color="normal",
        help="Signals that passed ML filter threshold"
    )

with col3:
    st.metric(
        "Blocked",
        ml_stats["blocked"],
        delta=f"-{100-ml_stats['pass_rate']:.1f}%" if ml_stats['total_signals'] > 0 else None,
        delta_color="inverse",
        help="Signals blocked by ML filter"
    )

with col4:
    pass_rate = ml_stats["pass_rate"]
    if pass_rate >= 70:
        color = "🟢"
    elif pass_rate >= 40:
        color = "🟡"
    else:
        color = "🔴"
    st.metric(
        "Pass Rate",
        f"{color} {pass_rate:.1f}%",
        help="Percentage of signals passing ML filter"
    )


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
# Footer
# ============================================================================

st.divider()
st.caption(
    "Pine Trading System Dashboard | "
    f"Monitoring last {log_window} minutes | "
    f"Auto-refresh: 10s | "
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
