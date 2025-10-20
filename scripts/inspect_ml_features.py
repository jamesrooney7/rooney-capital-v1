#!/usr/bin/env python3
"""Inspect ML feature readiness from the heartbeat or live worker state."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from runner import LiveWorker, RuntimeConfig, load_runtime_config  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect ML feature readiness for each configured symbol.",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        help="Path to the runtime configuration file (defaults to $PINE_RUNTIME_CONFIG).",
    )
    parser.add_argument(
        "--heartbeat",
        dest="heartbeat_path",
        help="Override the heartbeat file path discovered from the runtime config.",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "heartbeat", "worker"),
        default="auto",
        help=(
            "Select data source: read the heartbeat file, instantiate the live worker, "
            "or attempt heartbeat first then fall back to the worker (default)."
        ),
    )
    parser.add_argument(
        "--show-features",
        action="store_true",
        help="Print the numeric snapshot for each symbol in addition to readiness details.",
    )
    return parser.parse_args(argv)


def _load_report_from_heartbeat(path: Path) -> Mapping[str, Mapping[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Heartbeat file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Heartbeat file is not valid JSON: {path}") from exc

    details = payload.get("details")
    if not isinstance(details, Mapping):
        raise ValueError("Heartbeat file does not contain a 'details' section")
    ml_features = details.get("ml_features")
    if not isinstance(ml_features, Mapping):
        raise ValueError("Heartbeat file does not contain ML feature telemetry")
    normalised: dict[str, Mapping[str, object]] = {}
    for symbol, snapshot in ml_features.items():
        if isinstance(snapshot, Mapping):
            normalised[str(symbol)] = snapshot
    return normalised


def _load_report_from_worker(config: RuntimeConfig) -> Mapping[str, Mapping[str, object]]:
    worker: LiveWorker | None = None
    try:
        worker = LiveWorker(config)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialise LiveWorker: {exc}") from exc
    try:
        tracker = getattr(worker, "ml_feature_tracker", None)
        if tracker is None:
            return {}
        tracker.refresh_all()
        report = tracker.readiness_report()
        return report
    finally:
        if worker is not None:
            for subscriber in getattr(worker, "subscribers", []):
                try:
                    subscriber.stop()
                except Exception:
                    pass
            queue_manager = getattr(worker, "queue_manager", None)
            if queue_manager is not None:
                try:
                    queue_manager.broadcast_shutdown()
                except Exception:
                    pass


def _format_missing(snapshot: Mapping[str, object]) -> str:
    missing = snapshot.get("missing_features")
    if not missing:
        return "—"
    if isinstance(missing, Mapping):
        missing = list(missing.values())
    return ", ".join(str(item) for item in missing if item)


def _format_ready(value: object) -> str:
    return "yes" if bool(value) else "no"


def _format_feature_value(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _print_report(report: Mapping[str, Mapping[str, object]], show_features: bool) -> None:
    if not report:
        print("No ML feature bundles registered.")
        return

    print("=== ML Feature Readiness ===\n")
    for symbol in sorted(report):
        snapshot = report[symbol]
        ready = _format_ready(snapshot.get("ready"))
        missing = _format_missing(snapshot)
        print(f"Symbol: {symbol}")
        print(f"  Ready: {ready}")
        print(f"  Missing features: {missing}")
        if show_features:
            features = snapshot.get("features")
            if isinstance(features, Mapping) and features:
                print("  Feature snapshot:")
                for key in sorted(features):
                    print(f"    {key}: {_format_feature_value(features[key])}")
            else:
                print("  Feature snapshot: —")
        print()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        config = load_runtime_config(args.config_path)
    except Exception as exc:
        print(f"Error: failed to load runtime configuration: {exc}", file=sys.stderr)
        return 1

    heartbeat_report: Mapping[str, Mapping[str, object]] | None = None
    if args.source in {"auto", "heartbeat"}:
        heartbeat_candidate = args.heartbeat_path or config.heartbeat_file
        if heartbeat_candidate:
            try:
                heartbeat_report = _load_report_from_heartbeat(Path(heartbeat_candidate))
            except Exception as exc:
                if args.source == "heartbeat":
                    print(f"Error: {exc}", file=sys.stderr)
                    return 1
                print(f"Warning: {exc}; falling back to LiveWorker", file=sys.stderr)
        elif args.source == "heartbeat":
            print("Error: heartbeat path not provided in CLI or config", file=sys.stderr)
            return 1

    if heartbeat_report is not None:
        _print_report(heartbeat_report, args.show_features)
        return 0

    if args.source == "heartbeat":
        print("Error: heartbeat data unavailable", file=sys.stderr)
        return 1

    try:
        report = _load_report_from_worker(config)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    _print_report(report, args.show_features)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
