"""Discord webhook notifications for trading alerts and system monitoring.

Sends formatted alerts to Discord channels via webhooks for:
- Trade executions (entries/exits)
- System health and errors
- Daily performance summaries
- Custom alerts
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Optional
from urllib import request
from urllib.error import URLError

logger = logging.getLogger(__name__)

# Use US/Central for CME futures market times
MARKET_TIMEZONE = ZoneInfo("America/Chicago")


class DiscordNotifier:
    """Send notifications to Discord via webhook."""

    def __init__(self, webhook_url: str):
        """Initialize Discord notifier.

        Args:
            webhook_url: Discord webhook URL (from channel settings)
        """
        self.webhook_url = webhook_url
        self.enabled = bool(webhook_url)

        if self.enabled:
            logger.info("Discord notifier initialized")
        else:
            logger.warning("Discord notifier disabled (no webhook URL)")

    def _send_webhook(
        self,
        content: Optional[str] = None,
        embeds: Optional[list[dict[str, Any]]] = None,
    ) -> bool:
        """Send a webhook message to Discord.

        Args:
            content: Plain text message content
            embeds: List of rich embed objects

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            return False

        payload = {}
        if content:
            payload["content"] = content
        if embeds:
            payload["embeds"] = embeds

        try:
            data = json.dumps(payload).encode("utf-8")
            req = request.Request(
                self.webhook_url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "RooneyCapitalBot/1.0",
                },
            )
            with request.urlopen(req, timeout=10) as response:
                if response.status == 204:
                    return True
                else:
                    logger.warning(f"Discord webhook returned status {response.status}")
                    return False
        except URLError as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Discord notification: {e}")
            return False

    def send_trade_entry(
        self,
        symbol: str,
        side: str,
        price: float,
        size: float,
        ibs: Optional[float] = None,
        ml_score: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """Send trade entry notification.

        Args:
            symbol: Trading symbol (e.g., "ES", "NQ")
            side: "long" or "short"
            price: Entry price
            size: Position size (contracts)
            ibs: IBS value at entry
            ml_score: ML filter score
            timestamp: Entry time (defaults to now in market timezone)

        Returns:
            True if sent successfully
        """
        if timestamp is None:
            timestamp = datetime.now(tz=MARKET_TIMEZONE)
        elif timestamp.tzinfo is not None:
            # Convert to market timezone for display
            timestamp = timestamp.astimezone(MARKET_TIMEZONE)
        else:
            # Assume UTC if naive datetime
            timestamp = timestamp.replace(tzinfo=ZoneInfo("UTC")).astimezone(MARKET_TIMEZONE)

        embed = {
            "title": f"🟢 Trade Entry: {symbol}",
            "color": 3066993,  # Green
            "fields": [
                {"name": "Symbol", "value": symbol, "inline": True},
                {"name": "Side", "value": side.upper(), "inline": True},
                {"name": "Size", "value": f"{size} contracts", "inline": True},
                {"name": "Entry Price", "value": f"${price:.2f}", "inline": True},
                {"name": "Time (CT)", "value": timestamp.strftime("%Y-%m-%d %H:%M:%S %Z"), "inline": True},
            ],
            "timestamp": timestamp.isoformat(),
        }

        if ibs is not None:
            embed["fields"].append(
                {"name": "IBS", "value": f"{ibs:.3f}", "inline": True}
            )

        if ml_score is not None:
            embed["fields"].append(
                {"name": "ML Score", "value": f"{ml_score:.3f}", "inline": True}
            )

        return self._send_webhook(embeds=[embed])

    def send_trade_exit(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        size: float,
        pnl: float,
        pnl_percent: float,
        exit_reason: str,
        ibs: Optional[float] = None,
        duration_hours: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """Send trade exit notification.

        Args:
            symbol: Trading symbol
            side: "long" or "short"
            entry_price: Entry price
            exit_price: Exit price
            size: Position size
            pnl: Profit/loss in dollars
            pnl_percent: P&L as percentage
            exit_reason: Reason for exit
            ibs: IBS value at exit
            duration_hours: Trade duration in hours
            timestamp: Exit time (defaults to now in market timezone)

        Returns:
            True if sent successfully
        """
        if timestamp is None:
            timestamp = datetime.now(tz=MARKET_TIMEZONE)
        elif timestamp.tzinfo is not None:
            # Convert to market timezone for display
            timestamp = timestamp.astimezone(MARKET_TIMEZONE)
        else:
            # Assume UTC if naive datetime
            timestamp = timestamp.replace(tzinfo=ZoneInfo("UTC")).astimezone(MARKET_TIMEZONE)

        # Color based on profitability
        color = 3066993 if pnl > 0 else 15158332  # Green if profit, red if loss
        emoji = "✅" if pnl > 0 else "❌"

        embed = {
            "title": f"{emoji} Trade Exit: {symbol}",
            "color": color,
            "fields": [
                {"name": "Symbol", "value": symbol, "inline": True},
                {"name": "Side", "value": side.upper(), "inline": True},
                {"name": "Size", "value": f"{size} contracts", "inline": True},
                {"name": "Entry Price", "value": f"${entry_price:.2f}", "inline": True},
                {"name": "Exit Price", "value": f"${exit_price:.2f}", "inline": True},
                {"name": "P&L", "value": f"**${pnl:.2f}** ({pnl_percent:+.2f}%)", "inline": True},
                {"name": "Time (CT)", "value": timestamp.strftime("%Y-%m-%d %H:%M:%S %Z"), "inline": True},
                {"name": "Exit Reason", "value": exit_reason, "inline": True},
            ],
            "timestamp": timestamp.isoformat(),
        }

        if ibs is not None:
            embed["fields"].append(
                {"name": "IBS at Exit", "value": f"{ibs:.3f}", "inline": True}
            )

        if duration_hours is not None:
            embed["fields"].append(
                {"name": "Duration", "value": f"{duration_hours:.1f} hours", "inline": True}
            )

        return self._send_webhook(embeds=[embed])

    def send_daily_summary(
        self,
        total_pnl: float,
        num_trades: int,
        win_rate: float,
        best_trade: float,
        worst_trade: float,
        symbols_traded: list[str],
        date: Optional[datetime] = None,
        profit_factor: Optional[float] = None,
        avg_pnl: Optional[float] = None,
    ) -> bool:
        """Send daily performance summary.

        Args:
            total_pnl: Total P&L for the day
            num_trades: Number of trades executed
            win_rate: Win rate percentage
            best_trade: Best trade P&L
            worst_trade: Worst trade P&L
            symbols_traded: List of symbols traded
            date: Date for summary (defaults to today)
            profit_factor: Gross profit / gross loss ratio (optional)
            avg_pnl: Average P&L per trade (optional)

        Returns:
            True if sent successfully
        """
        if date is None:
            date = datetime.now()

        color = 3066993 if total_pnl > 0 else 15158332 if total_pnl < 0 else 3447003  # Green/Red/Blue

        symbols_str = ", ".join(symbols_traded) if symbols_traded else "None"

        # Calculate average if not provided
        if avg_pnl is None and num_trades > 0:
            avg_pnl = total_pnl / num_trades

        fields = [
            {"name": "Total P&L", "value": f"**${total_pnl:.2f}**", "inline": True},
            {"name": "Trades", "value": str(num_trades), "inline": True},
            {"name": "Win Rate", "value": f"{win_rate:.1f}%", "inline": True},
        ]

        # Add profit factor if available
        if profit_factor is not None:
            pf_display = f"{profit_factor:.2f}x" if profit_factor != float('inf') else "∞"
            pf_emoji = "🟢" if profit_factor > 1.5 else "🟡" if profit_factor > 1.0 else "🔴"
            fields.append({"name": "Profit Factor", "value": f"{pf_emoji} {pf_display}", "inline": True})

        # Add average P&L if available
        if avg_pnl is not None:
            fields.append({"name": "Avg P&L", "value": f"${avg_pnl:.2f}", "inline": True})

        fields.extend([
            {"name": "Best Trade", "value": f"${best_trade:.2f}", "inline": True},
            {"name": "Worst Trade", "value": f"${worst_trade:.2f}", "inline": True},
            {"name": "Symbols", "value": symbols_str, "inline": False},
        ])

        embed = {
            "title": f"📊 Daily Summary - {date.strftime('%Y-%m-%d')}",
            "color": color,
            "fields": fields,
            "timestamp": datetime.now().isoformat(),
        }

        return self._send_webhook(embeds=[embed])

    def send_system_alert(
        self,
        title: str,
        message: str,
        alert_type: str = "info",
        fields: Optional[list[dict[str, Any]]] = None,
    ) -> bool:
        """Send system alert notification.

        Args:
            title: Alert title
            message: Alert message/description
            alert_type: "info", "warning", or "error"
            fields: Optional additional fields

        Returns:
            True if sent successfully
        """
        # Color based on alert type
        colors = {
            "info": 3447003,    # Blue
            "warning": 16776960,  # Orange
            "error": 15158332,   # Red
        }
        color = colors.get(alert_type, 3447003)

        # Emoji based on alert type
        emojis = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "🚨",
        }
        emoji = emojis.get(alert_type, "ℹ️")

        embed = {
            "title": f"{emoji} {title}",
            "description": message,
            "color": color,
            "timestamp": datetime.now().isoformat(),
        }

        if fields:
            embed["fields"] = fields

        return self._send_webhook(embeds=[embed])

    def send_error(
        self,
        error_message: str,
        component: str,
        details: Optional[str] = None,
    ) -> bool:
        """Send error notification.

        Args:
            error_message: Error message
            component: Component/module where error occurred
            details: Additional error details

        Returns:
            True if sent successfully
        """
        fields = [
            {"name": "Component", "value": component, "inline": True},
        ]

        if details:
            fields.append({"name": "Details", "value": details[:1024], "inline": False})

        return self.send_system_alert(
            title="System Error",
            message=error_message,
            alert_type="error",
            fields=fields,
        )

    def send_health_check(
        self,
        status: str,
        uptime: str,
        services: dict[str, bool],
        recent_activity: Optional[str] = None,
    ) -> bool:
        """Send system health check notification.

        Args:
            status: Overall status ("healthy", "degraded", "down")
            uptime: System uptime string
            services: Dictionary of service name to status
            recent_activity: Recent activity summary

        Returns:
            True if sent successfully
        """
        # Color based on status
        colors = {
            "healthy": 3066993,  # Green
            "degraded": 16776960,  # Orange
            "down": 15158332,  # Red
        }
        color = colors.get(status, 3447003)

        fields = [
            {"name": "Status", "value": status.upper(), "inline": True},
            {"name": "Uptime", "value": uptime, "inline": True},
        ]

        # Add service statuses
        for service_name, service_status in services.items():
            status_emoji = "✅" if service_status else "❌"
            fields.append({
                "name": service_name,
                "value": status_emoji,
                "inline": True,
            })

        if recent_activity:
            fields.append({
                "name": "Recent Activity",
                "value": recent_activity,
                "inline": False,
            })

        embed = {
            "title": "💓 System Health Check",
            "color": color,
            "fields": fields,
            "timestamp": datetime.now().isoformat(),
        }

        return self._send_webhook(embeds=[embed])

    def send_custom(
        self,
        content: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        fields: Optional[list[dict[str, Any]]] = None,
        color: int = 3447003,
    ) -> bool:
        """Send custom notification.

        Args:
            content: Plain text content
            title: Embed title
            description: Embed description
            fields: Embed fields
            color: Embed color (integer)

        Returns:
            True if sent successfully
        """
        embeds = None
        if title or description or fields:
            embed: dict[str, Any] = {"color": color}
            if title:
                embed["title"] = title
            if description:
                embed["description"] = description
            if fields:
                embed["fields"] = fields
            embed["timestamp"] = datetime.now().isoformat()
            embeds = [embed]

        return self._send_webhook(content=content, embeds=embeds)


# Global notifier instance (initialized by config)
_notifier: Optional[DiscordNotifier] = None


def init_notifier(webhook_url: str) -> DiscordNotifier:
    """Initialize global Discord notifier.

    Args:
        webhook_url: Discord webhook URL

    Returns:
        DiscordNotifier instance
    """
    global _notifier
    _notifier = DiscordNotifier(webhook_url)
    return _notifier


def get_notifier() -> Optional[DiscordNotifier]:
    """Get global Discord notifier instance.

    Returns:
        DiscordNotifier instance or None if not initialized
    """
    return _notifier
