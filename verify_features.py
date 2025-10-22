#!/usr/bin/env python3
"""
Verify which ML features are working vs not working.
Parses ML scoring logs to find missing features.
"""

import json
import os
import re
import subprocess
from collections import defaultdict

def get_all_features():
    """Extract all unique features from model files."""
    features = set()
    models_dir = "src/models"

    for filename in os.listdir(models_dir):
        if filename.endswith("_best.json"):
            filepath = os.path.join(models_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                if 'Features' in data:
                    features.update(data['Features'])

    return sorted(features)

def get_model_features_by_symbol():
    """Get features for each symbol from model files."""
    symbol_features = {}
    models_dir = "src/models"

    for filename in os.listdir(models_dir):
        if filename.endswith("_best.json"):
            symbol = filename.replace("_best.json", "")
            filepath = os.path.join(models_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                if 'Features' in data:
                    symbol_features[symbol] = data['Features']

    return symbol_features

def parse_ml_scoring_logs():
    """Parse journalctl logs to find ML scoring results showing missing features."""
    print("\n=== Fetching ML scoring logs ===")

    cmd = [
        "sudo", "journalctl",
        "-u", "pine-runner.service",
        "-n", "5000",
        "--no-pager"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        logs = result.stdout
    except Exception as e:
        print(f"Error fetching logs: {e}")
        return None, None

    # Pattern for ML scoring incomplete warnings
    # Example: ⚠️ GC ML INCOMPLETE | Only 16/30 features calculated (53.3%) | Missing: 6s_hourly_z_score, ...
    incomplete_pattern = re.compile(
        r'⚠️\s+(\w+)\s+ML INCOMPLETE.*Missing:\s+([^)]+?)(?:\s+\(\+\d+\s+more\))?$'
    )

    # Pattern for successful ML scoring
    # Example: 🤖 GC ML HOURLY | Score: 0.718 | Passed: False | Threshold: 0.650 | Features: 16/30
    scoring_pattern = re.compile(
        r'🤖\s+(\w+)\s+ML\s+\w+\s+\|.*Features:\s+(\d+)/(\d+)'
    )

    missing_by_symbol = defaultdict(set)
    feature_counts = {}

    for line in logs.split('\n'):
        # Check for incomplete warnings
        match = incomplete_pattern.search(line)
        if match:
            symbol = match.group(1)
            missing_features_str = match.group(2)
            # Parse comma-separated list
            missing_features = [f.strip() for f in missing_features_str.split(',')]
            # Remove ellipsis if present
            missing_features = [f for f in missing_features if f and f != '...']
            missing_by_symbol[symbol].update(missing_features)

        # Check for scoring lines to get feature counts
        match = scoring_pattern.search(line)
        if match:
            symbol = match.group(1)
            calculated = int(match.group(2))
            total = int(match.group(3))
            feature_counts[symbol] = (calculated, total)

    return missing_by_symbol, feature_counts

def main():
    print("=" * 80)
    print("ML FEATURE VERIFICATION - PARSING LIVE ML SCORING LOGS")
    print("=" * 80)

    # Get all features from model files
    all_features = get_all_features()
    print(f"\nTotal unique features across all models: {len(all_features)}")

    # Get features by symbol
    symbol_features = get_model_features_by_symbol()
    print(f"Symbols with ML models: {', '.join(sorted(symbol_features.keys()))}")

    # Parse logs for missing features
    missing_by_symbol, feature_counts = parse_ml_scoring_logs()

    if not missing_by_symbol and not feature_counts:
        print("\n⚠️  Could not find ML scoring logs.")
        print("\nPossible reasons:")
        print("1. Service hasn't scored any symbols yet (still warming up)")
        print("2. All features are working (no incomplete warnings)")
        print("\nCheck recent logs:")
        print("  sudo journalctl -u pine-runner.service -n 100 | grep -E '🤖|⚠️'")
        return

    # Aggregate all missing features across symbols
    all_missing_features = set()
    for symbol, missing in missing_by_symbol.items():
        all_missing_features.update(missing)

    # Determine working features (all features minus missing ones)
    working_features = sorted(set(all_features) - all_missing_features)
    not_working_features = sorted(all_missing_features)

    print("\n" + "=" * 80)
    print("ML SCORING RESULTS BY SYMBOL")
    print("=" * 80)
    for symbol in sorted(symbol_features.keys()):
        if symbol in feature_counts:
            calculated, total = feature_counts[symbol]
            pct = 100 * calculated / total if total > 0 else 0
            status = "✅" if calculated == total else "⚠️"
            print(f"{status} {symbol:4s}: {calculated:2d}/{total:2d} features ({pct:5.1f}%)")
            if symbol in missing_by_symbol:
                missing_count = len(missing_by_symbol[symbol])
                print(f"         Missing {missing_count} features")

    print("\n" + "=" * 80)
    print(f"✅ WORKING FEATURES ({len(working_features)} features)")
    print("=" * 80)
    if working_features:
        for i, feature in enumerate(working_features, 1):
            print(f"  {i:3d}. {feature}")
    else:
        print("  None found")

    print("\n" + "=" * 80)
    print(f"⚠️  NOT WORKING FEATURES ({len(not_working_features)} features)")
    print("=" * 80)
    if not_working_features:
        for i, feature in enumerate(not_working_features, 1):
            print(f"  {i:3d}. {feature}")
    else:
        print("  None found - All features working!")

    print("\n" + "=" * 80)
    print("DETAILED BREAKDOWN BY SYMBOL")
    print("=" * 80)
    for symbol in sorted(missing_by_symbol.keys()):
        missing = sorted(missing_by_symbol[symbol])
        print(f"\n{symbol} - Missing {len(missing)} features:")
        for feature in missing:
            print(f"  - {feature}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total unique features:     {len(all_features)}")
    print(f"✅ Working:                {len(working_features)} ({100*len(working_features)//len(all_features) if all_features else 0}%)")
    print(f"⚠️  Not working:            {len(not_working_features)} ({100*len(not_working_features)//len(all_features) if all_features else 0}%)")
    print("=" * 80)

    # Save detailed report
    with open("feature_verification_report.txt", "w") as f:
        f.write("ML FEATURE VERIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total features: {len(all_features)}\n")
        f.write(f"Working: {len(working_features)}\n")
        f.write(f"Not working: {len(not_working_features)}\n\n")

        f.write("WORKING FEATURES:\n")
        f.write("-" * 80 + "\n")
        for feature in working_features:
            f.write(f"{feature}\n")

        f.write("\n\nNOT WORKING FEATURES:\n")
        f.write("-" * 80 + "\n")
        for feature in not_working_features:
            f.write(f"{feature}\n")

        f.write("\n\nBY SYMBOL:\n")
        f.write("-" * 80 + "\n")
        for symbol in sorted(missing_by_symbol.keys()):
            missing = sorted(missing_by_symbol[symbol])
            f.write(f"\n{symbol} - Missing {len(missing)} features:\n")
            for feature in missing:
                f.write(f"  - {feature}\n")

    print(f"\nDetailed report saved to: feature_verification_report.txt")

if __name__ == "__main__":
    main()
