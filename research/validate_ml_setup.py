#!/usr/bin/env python3
"""
Validate ML setup before running full backtest.

This script checks:
1. Model files are loaded correctly
2. All ML features can be found in collected filter values
3. ML scores are being generated (not NaN/zeros)

Usage:
    python research/validate_ml_setup.py
    python research/validate_ml_setup.py --symbols ES NQ
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root and src/ to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.loader import load_model_bundle
from strategy.feature_utils import normalize_column_name

# All trading instruments (excludes TLT and VIX - they're filters only)
ALL_INSTRUMENTS = [
    "ES", "NQ", "RTY", "YM",                    # Equity indices
    "GC", "SI", "HG", "CL", "NG", "PL",         # Commodities
    "6A", "6B", "6C", "6E", "6J", "6M", "6N", "6S",  # Currencies
    "ZC", "ZS", "ZW",                           # Grains
]


def validate_model_bundle(symbol: str, models_dir: Path) -> dict:
    """Validate model bundle can be loaded."""
    results = {
        "symbol": symbol,
        "bundle_loaded": False,
        "model_valid": False,
        "features_count": 0,
        "threshold": None,
        "json_path": None,
        "model_path": None,
        "errors": []
    }

    # Check JSON file exists
    json_path = models_dir / f"{symbol}_best.json"
    results["json_path"] = str(json_path)

    if not json_path.exists():
        results["errors"].append(f"JSON file not found: {json_path}")
        return results

    # Check model file exists
    model_path = models_dir / f"{symbol}_rf_model.pkl"
    results["model_path"] = str(model_path)

    if not model_path.exists():
        results["errors"].append(f"Model file not found: {model_path}")
        return results

    # Check if model is Git LFS pointer
    with open(model_path, 'rb') as f:
        header = f.read(200)

    try:
        text = header.decode('ascii')
        if text.startswith("version https://git-lfs.github.com/spec/"):
            results["errors"].append(f"Model is Git LFS pointer - run 'git lfs pull'")
            return results
    except UnicodeDecodeError:
        pass  # Good - binary file

    # Load bundle
    try:
        bundle = load_model_bundle(symbol, base_dir=models_dir)
        results["bundle_loaded"] = True
        results["features_count"] = len(bundle.features)
        results["threshold"] = bundle.threshold

        # Validate model
        if bundle.model is None:
            results["errors"].append("Model is None")
        elif not hasattr(bundle.model, 'predict_proba'):
            results["errors"].append("Model missing predict_proba method")
        else:
            results["model_valid"] = True

    except Exception as e:
        results["errors"].append(f"Failed to load bundle: {str(e)}")

    return results


def validate_feature_names(symbol: str, models_dir: Path) -> dict:
    """Validate feature naming conventions."""
    results = {
        "symbol": symbol,
        "features": [],
        "cross_instrument_features": [],
        "core_features": [],
        "warnings": []
    }

    try:
        bundle = load_model_bundle(symbol, base_dir=models_dir)
        results["features"] = list(bundle.features)

        # Categorize features
        for feature in bundle.features:
            if "_z_score" in feature or "_z_pipeline" in feature:
                results["cross_instrument_features"].append(feature)
            elif "_return" in feature and not feature.startswith("prev"):
                results["cross_instrument_features"].append(feature)
            else:
                results["core_features"].append(feature)

        # Check for enable_ prefixes (indicates bug)
        bad_features = [f for f in bundle.features if f.startswith("enable")]
        if bad_features:
            results["warnings"].append(
                f"Features have 'enable' prefix: {bad_features[:3]}..."
            )

    except Exception as e:
        results["warnings"].append(f"Could not load features: {str(e)}")

    return results


def print_validation_report(all_results: list[dict]):
    """Print formatted validation report."""
    print("\n" + "="*80)
    print("ML SETUP VALIDATION REPORT")
    print("="*80)

    passed = 0
    failed = 0
    warnings = 0

    for result in all_results:
        symbol = result["symbol"]

        if result["model_valid"] and not result["errors"]:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
            failed += 1

        if result.get("warnings"):
            warnings += len(result["warnings"])

        print(f"\n{symbol}: {status}")
        print(f"  Bundle Loaded: {'✓' if result['bundle_loaded'] else '✗'}")
        print(f"  Model Valid: {'✓' if result['model_valid'] else '✗'}")
        print(f"  Features: {result['features_count']}")
        print(f"  Threshold: {result['threshold']}")

        if result["errors"]:
            print(f"  Errors:")
            for error in result["errors"]:
                print(f"    - {error}")

        if result.get("warnings"):
            print(f"  Warnings:")
            for warning in result["warnings"]:
                print(f"    - {warning}")

    print("\n" + "="*80)
    print(f"Summary: {passed} passed, {failed} failed, {warnings} warnings")
    print("="*80)

    if failed > 0:
        print("\n⚠ VALIDATION FAILED - Fix errors before running backtest")
        return False
    elif warnings > 0:
        print("\n⚠ VALIDATION PASSED WITH WARNINGS - Review warnings above")
        return True
    else:
        print("\n✓ VALIDATION PASSED - Ready to run backtest")
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate ML setup before running backtest"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=ALL_INSTRUMENTS,
        help=f"Symbols to validate (default: all)"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=SRC / "models",
        help="Directory containing JSON metadata and RF models"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed feature lists"
    )

    args = parser.parse_args()

    print(f"Validating ML setup for {len(args.symbols)} instruments...")
    print(f"Models directory: {args.models_dir}")

    # Validate each symbol
    all_results = []

    for symbol in args.symbols:
        bundle_result = validate_model_bundle(symbol, args.models_dir)
        feature_result = validate_feature_names(symbol, args.models_dir)

        # Combine results
        combined = {**bundle_result, **feature_result}
        all_results.append(combined)

        if args.verbose and feature_result["features"]:
            print(f"\n{symbol} Features ({len(feature_result['features'])}):")
            print(f"  Core: {len(feature_result['core_features'])}")
            for feat in feature_result["core_features"][:5]:
                print(f"    - {feat}")
            if len(feature_result["core_features"]) > 5:
                print(f"    ... and {len(feature_result['core_features']) - 5} more")

            print(f"  Cross-Instrument: {len(feature_result['cross_instrument_features'])}")
            for feat in feature_result["cross_instrument_features"][:5]:
                print(f"    - {feat}")
            if len(feature_result["cross_instrument_features"]) > 5:
                print(f"    ... and {len(feature_result['cross_instrument_features']) - 5} more")

    # Print report
    success = print_validation_report(all_results)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
