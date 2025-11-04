#!/usr/bin/env python3
"""
Fix missing Prod_Threshold field in ML model metadata files.

These 5 instruments have "threshold" but not "Prod_Threshold" which the
loader code requires.
"""
import json
from pathlib import Path

# Models that need fixing
SYMBOLS = ["6C", "6M", "6N", "6S", "PL"]
MODELS_DIR = Path(__file__).parent.parent / "src" / "models"

def fix_model_threshold(symbol: str) -> None:
    """Add Prod_Threshold field to model metadata if missing."""

    json_path = MODELS_DIR / f"{symbol}_best.json"

    if not json_path.exists():
        print(f"❌ {symbol}: File not found at {json_path}")
        return

    # Load existing JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Check if Prod_Threshold already exists
    if "Prod_Threshold" in data:
        print(f"✓ {symbol}: Prod_Threshold already exists ({data['Prod_Threshold']})")
        return

    # Check if lowercase threshold exists
    if "threshold" not in data:
        print(f"⚠️  {symbol}: No threshold field found in JSON!")
        return

    threshold_value = data["threshold"]

    # Add Prod_Threshold field (insert after symbol for readability)
    # We'll rebuild the dict with Prod_Threshold in the right position
    new_data = {}
    for key, value in data.items():
        new_data[key] = value
        if key == "symbol":
            new_data["Prod_Threshold"] = threshold_value

    # Write back to file with nice formatting
    with open(json_path, 'w') as f:
        json.dump(new_data, f, indent=2)

    print(f"✓ {symbol}: Added Prod_Threshold={threshold_value}")

def main():
    print("=" * 70)
    print("🔧 FIXING MODEL THRESHOLD FIELDS")
    print("=" * 70)
    print()
    print(f"Models directory: {MODELS_DIR}")
    print(f"Symbols to fix: {', '.join(SYMBOLS)}")
    print()

    for symbol in SYMBOLS:
        fix_model_threshold(symbol)

    print()
    print("=" * 70)
    print("✅ COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Restart pine-runner: sudo systemctl restart pine-runner")
    print("  2. Verify all 16 models load: sudo journalctl -u pine-runner -f | grep 'Model loaded'")
    print()

if __name__ == "__main__":
    main()
