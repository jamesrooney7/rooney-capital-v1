"""Utility functions for parsing logs and reading system status."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

# Paths
HEARTBEAT_PATH = Path("/var/run/pine/worker_heartbeat.json")
SERVICE_NAME = "pine-runner.service"


def read_heartbeat() -> dict[str, Any]:
    """Read the worker heartbeat file.

    Returns:
        Dictionary with status, updated_at, and details.
        Returns empty dict with status='unknown' if file doesn't exist.
    """
    if not HEARTBEAT_PATH.exists():
        return {
            "status": "unknown",
            "updated_at": None,
            "details": {},
        }

    try:
        with open(HEARTBEAT_PATH, "r") as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, IOError):
        return {
            "status": "error",
            "updated_at": None,
            "details": {},
        }


def get_service_status() -> dict[str, Any]:
    """Get systemd service status.

    Returns:
        Dictionary with is_active (bool), is_running (bool), uptime (str).
    """
    try:
        # Check if service is active
        result = subprocess.run(
            ["systemctl", "is-active", SERVICE_NAME],
            capture_output=True,
            text=True,
            timeout=5,
        )
        is_active = result.returncode == 0

        # Get service details
        result = subprocess.run(
            ["systemctl", "show", SERVICE_NAME, "--no-pager"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        props = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                props[key] = value

        # Parse active state and uptime
        active_state = props.get("ActiveState", "unknown")
        active_enter = props.get("ActiveEnterTimestamp", "")

        uptime = "Unknown"
        if active_enter:
            try:
                start_time = datetime.strptime(
                    active_enter.split()[0] + " " + active_enter.split()[1],
                    "%a %Y-%m-%d"
                )
                uptime_delta = datetime.now() - start_time
                days = uptime_delta.days
                hours, remainder = divmod(uptime_delta.seconds, 3600)
                minutes, _ = divmod(remainder, 60)

                if days > 0:
                    uptime = f"{days}d {hours}h {minutes}m"
                elif hours > 0:
                    uptime = f"{hours}h {minutes}m"
                else:
                    uptime = f"{minutes}m"
            except (ValueError, IndexError):
                pass

        return {
            "is_active": is_active,
            "is_running": active_state == "active",
            "uptime": uptime,
            "state": active_state,
        }
    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
        return {
            "is_active": False,
            "is_running": False,
            "uptime": "Unknown",
            "state": "unknown",
        }


def parse_recent_logs(
    lines: int = 1000,
    since_minutes: int = 60,
) -> dict[str, Any]:
    """Parse recent journalctl logs for trading activity.

    Args:
        lines: Number of recent log lines to fetch
        since_minutes: Only parse logs from last N minutes

    Returns:
        Dictionary with:
        - entries: list of entry signals
        - exits: list of exit signals
        - fills: list of order fills
        - ml_blocked: list of ML-blocked trades
        - errors: list of error messages
    """
    try:
        # Fetch recent logs
        result = subprocess.run(
            [
                "journalctl",
                "-u", SERVICE_NAME,
                "-n", str(lines),
                "--no-pager",
                "--output=short-iso",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return _empty_log_result()

        logs = result.stdout

    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
        return _empty_log_result()

    # Parse logs
    entries = []
    exits = []
    fills = []
    ml_blocked = []
    errors = []

    # Cutoff time (timezone-aware to match log timestamps)
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=since_minutes)

    for line in logs.splitlines():
        # Extract timestamp
        timestamp = _extract_timestamp(line)
        if timestamp and timestamp < cutoff:
            continue

        # Entry signals: "ES ENTRY | IBS: 0.081 | ML Score: 0.723 | Passed: True"
        entry_match = re.search(
            r"(\w+)\s+ENTRY\s+\|\s+IBS:\s+([\d.]+)\s+\|\s+ML Score:\s+([\d.]+)\s+\|\s+Passed:\s+(\w+)",
            line
        )
        if entry_match:
            entries.append({
                "timestamp": timestamp,
                "symbol": entry_match.group(1),
                "ibs": float(entry_match.group(2)),
                "ml_score": float(entry_match.group(3)),
                "passed": entry_match.group(4) == "True",
                "raw": line,
            })
            continue

        # Fills: "ES BUY FILLED | Price: 5812.25 | Size: 1"
        fill_match = re.search(
            r"(\w+)\s+(BUY|SELL)\s+FILLED\s+\|\s+Price:\s+([\d.]+)\s+\|\s+Size:\s+([\d.]+)",
            line
        )
        if fill_match:
            fills.append({
                "timestamp": timestamp,
                "symbol": fill_match.group(1),
                "action": fill_match.group(2).lower(),
                "price": float(fill_match.group(3)),
                "size": float(fill_match.group(4)),
                "raw": line,
            })
            continue

        # Exits: "ES EXIT | IBS: 0.850 | Reason: ibs_high"
        exit_match = re.search(
            r"(\w+)\s+EXIT\s+\|\s+IBS:\s+([\d.]+)\s+\|\s+Reason:\s+(\w+)",
            line
        )
        if exit_match:
            exits.append({
                "timestamp": timestamp,
                "symbol": exit_match.group(1),
                "ibs": float(exit_match.group(2)),
                "reason": exit_match.group(3),
                "raw": line,
            })
            continue

        # ML blocked: "⛔ ES ML FILTER BLOCKED ENTRY | Score 0.423 < Threshold 0.5"
        blocked_match = re.search(
            r"⛔\s+(\w+)\s+ML FILTER BLOCKED ENTRY\s+\|\s+Score\s+([\d.]+)\s+<\s+Threshold\s+([\d.]+)",
            line
        )
        if blocked_match:
            ml_blocked.append({
                "timestamp": timestamp,
                "symbol": blocked_match.group(1),
                "ml_score": float(blocked_match.group(2)),
                "threshold": float(blocked_match.group(3)),
                "raw": line,
            })
            continue

        # Errors
        if "ERROR" in line.upper() or "FAILED" in line.upper():
            errors.append({
                "timestamp": timestamp,
                "raw": line,
            })

    return {
        "entries": entries,
        "exits": exits,
        "fills": fills,
        "ml_blocked": ml_blocked,
        "errors": errors,
    }


def _empty_log_result() -> dict[str, list]:
    """Return empty log result structure."""
    return {
        "entries": [],
        "exits": [],
        "fills": [],
        "ml_blocked": [],
        "errors": [],
    }


def _extract_timestamp(log_line: str) -> Optional[datetime]:
    """Extract timestamp from ISO format log line.

    Args:
        log_line: Log line starting with ISO timestamp

    Returns:
        Datetime object or None if parsing fails
    """
    # Format: 2025-10-24T01:23:45+0000
    match = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{4})", log_line)
    if not match:
        return None

    try:
        return datetime.strptime(match.group(1), "%Y-%m-%dT%H:%M:%S%z")
    except ValueError:
        return None


def calculate_ml_stats(log_data: dict[str, Any]) -> dict[str, Any]:
    """Calculate ML filter statistics from parsed logs.

    Args:
        log_data: Result from parse_recent_logs()

    Returns:
        Dictionary with total_signals, passed, blocked, pass_rate
    """
    total_passed = len(log_data["entries"])
    total_blocked = len(log_data["ml_blocked"])
    total_signals = total_passed + total_blocked

    pass_rate = (total_passed / total_signals * 100) if total_signals > 0 else 0.0

    return {
        "total_signals": total_signals,
        "passed": total_passed,
        "blocked": total_blocked,
        "pass_rate": pass_rate,
    }


def get_recent_trades(log_data: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:
    """Extract recent completed trades from log data.

    Args:
        log_data: Result from parse_recent_logs()
        limit: Maximum number of trades to return

    Returns:
        List of trade dictionaries with entry and exit info
    """
    # Match fills to entries/exits by symbol and timestamp proximity
    trades = []

    for fill in log_data["fills"]:
        if fill["action"] == "buy":
            # This is an entry - find matching entry signal
            entry_signal = _find_matching_entry(fill, log_data["entries"])

            trades.append({
                "type": "entry",
                "timestamp": fill["timestamp"],
                "symbol": fill["symbol"],
                "action": fill["action"],
                "price": fill["price"],
                "size": fill["size"],
                "ibs": entry_signal.get("ibs") if entry_signal else None,
                "ml_score": entry_signal.get("ml_score") if entry_signal else None,
            })

        elif fill["action"] == "sell":
            # This is an exit - find matching exit signal
            exit_signal = _find_matching_exit(fill, log_data["exits"])

            trades.append({
                "type": "exit",
                "timestamp": fill["timestamp"],
                "symbol": fill["symbol"],
                "action": fill["action"],
                "price": fill["price"],
                "size": fill["size"],
                "ibs": exit_signal.get("ibs") if exit_signal else None,
                "reason": exit_signal.get("reason") if exit_signal else None,
            })

    # Sort by timestamp descending and limit
    trades.sort(key=lambda x: x["timestamp"] or datetime.min, reverse=True)
    return trades[:limit]


def _find_matching_entry(
    fill: dict[str, Any],
    entries: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Find entry signal matching a buy fill."""
    # Find entry with same symbol within 5 minutes before fill
    fill_time = fill["timestamp"]
    if not fill_time:
        return None

    for entry in entries:
        if entry["symbol"] != fill["symbol"]:
            continue
        if not entry["timestamp"]:
            continue

        time_diff = (fill_time - entry["timestamp"]).total_seconds()
        if 0 <= time_diff <= 300:  # Within 5 minutes
            return entry

    return None


def _find_matching_exit(
    fill: dict[str, Any],
    exits: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Find exit signal matching a sell fill."""
    # Find exit with same symbol within 5 minutes before fill
    fill_time = fill["timestamp"]
    if not fill_time:
        return None

    for exit_sig in exits:
        if exit_sig["symbol"] != fill["symbol"]:
            continue
        if not exit_sig["timestamp"]:
            continue

        time_diff = (fill_time - exit_sig["timestamp"]).total_seconds()
        if 0 <= time_diff <= 300:  # Within 5 minutes
            return exit_sig

    return None


def get_open_positions(log_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Determine currently open positions from log data.

    Args:
        log_data: Result from parse_recent_logs()

    Returns:
        List of open position dictionaries
    """
    # Track net position per symbol
    positions: dict[str, dict[str, Any]] = {}

    for fill in log_data["fills"]:
        symbol = fill["symbol"]

        if symbol not in positions:
            positions[symbol] = {
                "symbol": symbol,
                "size": 0.0,
                "entry_price": None,
                "entry_time": None,
            }

        if fill["action"] == "buy":
            positions[symbol]["size"] += fill["size"]
            positions[symbol]["entry_price"] = fill["price"]
            positions[symbol]["entry_time"] = fill["timestamp"]
        elif fill["action"] == "sell":
            positions[symbol]["size"] -= fill["size"]

    # Filter to only open positions (size > 0)
    open_positions = [
        pos for pos in positions.values()
        if pos["size"] > 0
    ]

    return open_positions
