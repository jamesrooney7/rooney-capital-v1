#!/usr/bin/env python3
"""
End-to-End Test of Portfolio Optimization System

Tests the complete workflow:
1. Generate sample data
2. Run portfolio optimizer
3. Test risk manager
4. Validate outputs

Author: Rooney Capital
Date: 2025-01-22
"""

import subprocess
import json
from pathlib import Path
import sys

def run_command(cmd, description):
    """Run command and report success/failure."""
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ SUCCESS: {description}")
        return True
    else:
        print(f"❌ FAILED: {description}")
        print(f"Error: {result.stderr}")
        return False


def main():
    print("="*80)
    print("END-TO-END TEST: Portfolio Optimization System")
    print("="*80)

    test_results = []

    # Test 1: Generate sample data
    test_results.append(run_command(
        ['python', 'research/portfolio_optimization/create_sample_data.py'],
        "Generate sample validation data"
    ))

    # Test 2: Run portfolio optimizer
    test_results.append(run_command(
        [
            'python', 'research/portfolio_optimization/portfolio_optimizer.py',
            '--validation-dir', 'test_data/ml_meta_labeling/results',
            '--max-drawdown', '6000',
            '--daily-loss-limit', '3000',
            '--output', 'test_data/portfolio_manifest.json'
        ],
        "Run portfolio optimizer"
    ))

    # Test 3: Validate manifest output
    print(f"\n{'='*80}")
    print("TEST: Validate portfolio manifest")
    print(f"{'='*80}\n")

    manifest_path = Path('test_data/portfolio_manifest.json')
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

        print(f"Manifest version: {manifest['version']}")
        print(f"Strategies selected: {len(manifest['selected_strategies'])}")
        print(f"\n2022-2023 Optimization:")
        print(f"  Sharpe: {manifest['performance']['optimization_period']['sharpe_ratio']:.2f}")
        print(f"  Max DD: ${manifest['performance']['optimization_period']['max_drawdown']:,.0f}")
        print(f"  Max Daily Loss: ${manifest['performance']['optimization_period']['max_daily_loss']:,.0f}")

        print(f"\n2024 Out-of-Sample:")
        print(f"  Sharpe: {manifest['performance']['test_period']['sharpe_ratio']:.2f}")
        print(f"  Max DD: ${manifest['performance']['test_period']['max_drawdown']:,.0f}")
        print(f"  Max Daily Loss: ${manifest['performance']['test_period']['max_daily_loss']:,.0f}")

        # Validate constraints
        opt_dd = manifest['performance']['optimization_period']['max_drawdown']
        opt_daily = manifest['performance']['optimization_period']['max_daily_loss']

        if opt_dd <= 6000 and opt_daily <= 3000:
            print(f"\n✅ Optimization period meets constraints")
            test_results.append(True)
        else:
            print(f"\n❌ Optimization period VIOLATES constraints!")
            test_results.append(False)
    else:
        print("❌ Manifest file not found!")
        test_results.append(False)

    # Test 4: Test risk manager
    test_results.append(run_command(
        ['python', 'research/portfolio_optimization/risk_manager.py'],
        "Test risk manager with demo trades"
    ))

    # Summary
    print(f"\n\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests: {len(test_results)}")
    print(f"Passed: {sum(test_results)}")
    print(f"Failed: {len(test_results) - sum(test_results)}")

    if all(test_results):
        print(f"\n🎉 ALL TESTS PASSED! Portfolio optimization system is working correctly.")
        return 0
    else:
        print(f"\n⚠️  SOME TESTS FAILED. Please review the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
