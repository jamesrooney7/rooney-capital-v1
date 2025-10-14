#!/usr/bin/env python3
"""Launch the live worker with the current runtime configuration."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from runner import LiveWorker, load_runtime_config  # noqa: E402


def main() -> int:
    print("=== Rooney Capital Worker Starting ===\n")

    try:
        config = load_runtime_config()
    except Exception as exc:  # pragma: no cover - defensive runtime logging
        print(f"✗ Failed to load runtime configuration: {exc}")
        return 1

    if config.killswitch:
        print("⚠️  POLICY_KILLSWITCH is enabled")
        print("    No orders will be sent - paper trading mode\n")
    else:
        print("🚨 POLICY_KILLSWITCH is disabled")
        print("    Live orders will be transmitted\n")

    print(f"Trading symbols: {', '.join(config.symbols)}")
    print(f"Starting cash: ${config.starting_cash:,.0f}")
    if config.heartbeat_file:
        print(f"Heartbeat file: {config.heartbeat_file}")
    print()

    try:
        worker = LiveWorker(config)
    except Exception as exc:  # pragma: no cover - defensive runtime logging
        print(f"✗ Failed to initialise worker: {exc}")
        return 1

    print("Starting worker (Ctrl+C to stop)…\n")
    try:
        worker.run()
    except KeyboardInterrupt:  # pragma: no cover - runtime convenience
        print("\nShutdown requested by user")
        return 0
    except Exception as exc:  # pragma: no cover - defensive runtime logging
        print(f"\nFATAL ERROR: {exc}")
        import traceback

        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
