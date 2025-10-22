#!/usr/bin/env python3
"""
Verify which ML features are working vs not working.
Parses logs to find feature collection output and categorizes features.
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

def parse_log_for_features(symbol="ES"):
    """Parse journalctl logs to find feature values for a symbol."""
    print(f"\n=== Fetching logs for {symbol} ===")

    # Get recent logs with ML feature output
    cmd = [
        "sudo", "journalctl",
        "-u", "pine-runner.service",
        "-n", "2000",
        "--no-pager"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        logs = result.stdout
    except Exception as e:
        print(f"Error fetching logs: {e}")
        return None

    # Look for ML feature collection output
    # Format: "  feature_name: value" or "  feature_name: None"
    feature_pattern = re.compile(r'^\s{2,}([a-z_0-9]+):\s+(.+)$', re.MULTILINE)

    # Find the most recent feature snapshot for this symbol
    lines = logs.split('\n')
    feature_values = {}
    in_feature_block = False
    current_symbol = None

    for line in lines:
        # Check if we're entering a feature collection block
        if f"ML features for {symbol}" in line or f"collect_filter_values {symbol}" in line:
            in_feature_block = True
            current_symbol = symbol
            continue

        # Check if we're leaving the block
        if in_feature_block and (line.strip() == "" or "ML features for" in line):
            if "ML features for" in line and symbol not in line:
                in_feature_block = False
                current_symbol = None

        # Parse feature lines
        if in_feature_block:
            match = feature_pattern.match(line)
            if match:
                feature_name = match.group(1)
                feature_value = match.group(2).strip()
                feature_values[feature_name] = feature_value

    return feature_values

def categorize_features(all_features, log_features):
    """Categorize features into working vs not working."""
    working = []
    not_working = []
    missing = []

    for feature in all_features:
        if feature in log_features:
            value = log_features[feature]
            if value == "None" or value == "nan" or value == "null":
                not_working.append(feature)
            else:
                working.append(feature)
        else:
            missing.append(feature)

    return working, not_working, missing

def main():
    print("=" * 80)
    print("ML FEATURE VERIFICATION")
    print("=" * 80)

    # Get all features from model files
    all_features = get_all_features()
    print(f"\nTotal features across all models: {len(all_features)}")

    # Parse logs for feature values (checking ES as representative symbol)
    log_features = parse_log_for_features("ES")

    if not log_features:
        print("\n⚠️  Could not find feature values in logs.")
        print("\nTry one of these approaches:")
        print("1. Check if the service is running: sudo systemctl status pine-runner.service")
        print("2. Look for recent feature output: sudo journalctl -u pine-runner.service -n 500 | grep -A 50 'ML features'")
        print("3. Restart service to trigger feature collection: sudo systemctl restart pine-runner.service")
        return

    print(f"Found {len(log_features)} feature values in logs")

    # Categorize features
    working, not_working, missing = categorize_features(all_features, log_features)

    print("\n" + "=" * 80)
    print(f"✅ WORKING FEATURES ({len(working)} features)")
    print("=" * 80)
    for feature in working:
        value = log_features.get(feature, "N/A")
        # Truncate long values
        if len(value) > 50:
            value = value[:47] + "..."
        print(f"  {feature:45s} = {value}")

    print("\n" + "=" * 80)
    print(f"⚠️  NOT WORKING (None/null values) ({len(not_working)} features)")
    print("=" * 80)
    for feature in not_working:
        print(f"  {feature}")

    if missing:
        print("\n" + "=" * 80)
        print(f"❓ MISSING FROM LOGS ({len(missing)} features)")
        print("=" * 80)
        for feature in missing:
            print(f"  {feature}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total features:      {len(all_features)}")
    print(f"✅ Working:          {len(working)} ({100*len(working)//len(all_features)}%)")
    print(f"⚠️  Not working:      {len(not_working)} ({100*len(not_working)//len(all_features)}%)")
    if missing:
        print(f"❓ Missing:          {len(missing)} ({100*len(missing)//len(all_features)}%)")
    print("=" * 80)

    # Save detailed report
    with open("feature_verification_report.txt", "w") as f:
        f.write("ML FEATURE VERIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total features: {len(all_features)}\n")
        f.write(f"Working: {len(working)}\n")
        f.write(f"Not working: {len(not_working)}\n")
        f.write(f"Missing: {len(missing)}\n\n")

        f.write("WORKING FEATURES:\n")
        f.write("-" * 80 + "\n")
        for feature in working:
            f.write(f"{feature}: {log_features.get(feature, 'N/A')}\n")

        f.write("\n\nNOT WORKING FEATURES:\n")
        f.write("-" * 80 + "\n")
        for feature in not_working:
            f.write(f"{feature}\n")

        if missing:
            f.write("\n\nMISSING FEATURES:\n")
            f.write("-" * 80 + "\n")
            for feature in missing:
                f.write(f"{feature}\n")

    print(f"\nDetailed report saved to: feature_verification_report.txt")

if __name__ == "__main__":
    main()
