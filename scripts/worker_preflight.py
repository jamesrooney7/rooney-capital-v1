#!/usr/bin/env python3
"""Run LiveWorker pre-flight checks using the current runtime configuration."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from runner import LiveWorker, load_runtime_config  # noqa: E402


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def main() -> int:
    print("=== Rooney Capital Worker Preflight ===\n")

    try:
        config = load_runtime_config()
    except Exception as exc:  # pragma: no cover - defensive runtime logging
        print(f"✗ Failed to load runtime configuration: {exc}")
        return 1

    print("Configuration summary:")
    print(f"  Symbols: {', '.join(config.symbols) if config.symbols else '—'}")
    print(f"  Starting cash: ${config.starting_cash:,.0f}")
    if config.models_path:
        print(f"  Models path: {config.models_path}")
    print(f"  Killswitch: {_format_bool(config.killswitch)}\n")

    if config.killswitch:
        print("✓ POLICY_KILLSWITCH is enabled (paper trading mode)")
    else:
        print("⚠️  POLICY_KILLSWITCH is disabled (live trading mode)")

    print("\nInitialising worker…")
    try:
        worker = LiveWorker(config)
    except Exception as exc:  # pragma: no cover - defensive runtime logging
        print(f"✗ Failed to initialise worker: {exc}")
        return 1

    print("✓ Worker created")
    print(f"  Trading symbols: {', '.join(worker.symbols)}")
    print(f"  Reference feeds: {', '.join(worker.reference_symbols)}")
    print(f"  Total data feeds: {len(worker.data_symbols)}\n")

    print("Running pre-flight checks…")
    try:
        passed = worker.run_preflight_checks()
    except Exception as exc:  # pragma: no cover - defensive runtime logging
        print(f"✗ Pre-flight checks raised an error: {exc}")
        return 1

    if not passed:
        print("✗ Pre-flight checks reported failures. Review log output above for details.")
        return 1

    print("\n=== Preflight Complete ===")
    print("✓ Worker passed all checks and is ready for launch.")
    print("\nTo start the worker run: python scripts/launch_worker.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
