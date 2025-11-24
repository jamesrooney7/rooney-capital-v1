#!/usr/bin/env python3
"""
Complete End-to-End Pipeline Test

Tests all phases of the automated trading strategy pipeline:
1. Phase 1: Strategy Factory (sample database creation)
2. Phase 2: Extract Winners
3. Phase 3: ML Meta-Labeling (validation data already created)
4. Phase 4: Portfolio Optimization

This validates that the entire automation pipeline works correctly
before deploying to production.

Author: Rooney Capital
Date: 2025-01-22
"""

import subprocess
import json
import sqlite3
from pathlib import Path
import sys

def print_header(title):
    """Print formatted header."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")


def run_command(cmd, description, check=True):
    """Run command and report success/failure."""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0 or not check:
        print(f"✅ SUCCESS: {description}")
        if result.stdout:
            # Print last 20 lines of output
            lines = result.stdout.split('\n')
            relevant_lines = [l for l in lines if l.strip()][-20:]
            for line in relevant_lines:
                print(f"  {line}")
        return True
    else:
        print(f"❌ FAILED: {description}")
        print(f"Error: {result.stderr}")
        return False


def validate_database(db_path):
    """Validate Strategy Factory database structure and content."""
    print_header("VALIDATING PHASE 1: Strategy Factory Database")

    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return False

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check execution_runs table
    cursor.execute("SELECT COUNT(*) FROM execution_runs WHERE phase = 1 AND status = 'completed'")
    run_count = cursor.fetchone()[0]
    print(f"  Completed Phase 1 runs: {run_count}")

    # Check backtest_results table
    cursor.execute("SELECT COUNT(*) FROM backtest_results")
    result_count = cursor.fetchone()[0]
    print(f"  Total backtest results: {result_count}")

    # Check strategy distribution
    cursor.execute("""
        SELECT strategy_name, COUNT(*) as count
        FROM backtest_results
        GROUP BY strategy_name
        ORDER BY count DESC
        LIMIT 5
    """)
    top_strategies = cursor.fetchall()
    print(f"\n  Top strategies:")
    for name, count in top_strategies:
        print(f"    {name}: {count} results")

    conn.close()

    if run_count >= 15 and result_count >= 100:  # Should have runs for all 15 instruments
        print(f"\n✅ Database validation passed")
        return True
    else:
        print(f"\n❌ Database validation failed")
        return False


def validate_winners_manifest(manifest_path):
    """Validate winners manifest structure and content."""
    print_header("VALIDATING PHASE 2: Winners Manifest")

    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        return False

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Check structure
    required_keys = ['version', 'created_at', 'total_winners', 'winners_by_instrument', 'winners']
    missing_keys = [k for k in required_keys if k not in manifest]

    if missing_keys:
        print(f"❌ Missing required keys: {missing_keys}")
        return False

    print(f"  Version: {manifest['version']}")
    print(f"  Total winners: {manifest['total_winners']}")
    print(f"  Instruments: {len(manifest['winners_by_instrument'])}")

    # Check winners
    if len(manifest['winners']) > 0:
        sample_winner = manifest['winners'][0]
        print(f"\n  Sample winner:")
        print(f"    Symbol: {sample_winner['symbol']}")
        print(f"    Strategy: {sample_winner['strategy_name']} (ID {sample_winner['strategy_id']})")
        print(f"    Sharpe: {sample_winner['sharpe_ratio']:.2f}")
        print(f"    Trades: {sample_winner['total_trades']}")

    if manifest['total_winners'] >= 50:  # Should have multiple winners per instrument
        print(f"\n✅ Winners manifest validation passed")
        return True
    else:
        print(f"\n❌ Winners manifest validation failed")
        return False


def validate_ml_validation_data(validation_dir):
    """Validate ML validation data exists for strategies."""
    print_header("VALIDATING PHASE 3: ML Validation Data")

    if not validation_dir.exists():
        print(f"❌ Validation directory not found: {validation_dir}")
        return False

    strategy_dirs = [d for d in validation_dir.iterdir() if d.is_dir()]
    print(f"  Found {len(strategy_dirs)} strategy directories")

    valid_strategies = 0
    for strat_dir in strategy_dirs[:5]:  # Check first 5
        equity_files = list(strat_dir.glob('*_validation_equity.csv'))
        if equity_files:
            print(f"  ✅ {strat_dir.name}: Has equity curve")
            valid_strategies += 1
        else:
            print(f"  ⚠️  {strat_dir.name}: No equity curve found")

    if valid_strategies >= 3:
        print(f"\n✅ ML validation data present ({valid_strategies} strategies)")
        return True
    else:
        print(f"\n❌ Insufficient validation data")
        return False


def validate_portfolio_manifest(manifest_path):
    """Validate portfolio optimization manifest."""
    print_header("VALIDATING PHASE 4: Portfolio Manifest")

    if not manifest_path.exists():
        print(f"❌ Portfolio manifest not found: {manifest_path}")
        return False

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Check structure
    required_keys = ['version', 'selected_strategies', 'performance', 'risk_management']
    missing_keys = [k for k in required_keys if k not in manifest]

    if missing_keys:
        print(f"❌ Missing required keys: {missing_keys}")
        return False

    print(f"  Version: {manifest['version']}")
    print(f"  Selected strategies: {len(manifest['selected_strategies'])}")

    # Check performance
    if 'optimization_period' in manifest['performance']:
        opt_perf = manifest['performance']['optimization_period']
        print(f"\n  Optimization Period (2022-2023):")
        print(f"    Sharpe Ratio: {opt_perf['sharpe_ratio']:.2f}")
        print(f"    Max Drawdown: ${opt_perf['max_drawdown']:,.0f}")
        print(f"    Max Daily Loss: ${opt_perf['max_daily_loss']:,.0f}")

    if 'test_period' in manifest['performance']:
        test_perf = manifest['performance']['test_period']
        print(f"\n  Test Period (2024):")
        print(f"    Sharpe Ratio: {test_perf['sharpe_ratio']:.2f}")
        print(f"    Max Drawdown: ${test_perf['max_drawdown']:,.0f}")
        print(f"    Max Daily Loss: ${test_perf['max_daily_loss']:,.0f}")

    # Check constraints
    if 'optimization_period' in manifest['performance']:
        opt_dd = manifest['performance']['optimization_period']['max_drawdown']
        opt_daily = manifest['performance']['optimization_period']['max_daily_loss']
        dd_limit = manifest['optimization_config']['max_drawdown']
        daily_limit = manifest['optimization_config']['daily_loss_limit']

        print(f"\n  Constraint Validation:")
        dd_ok = opt_dd <= dd_limit
        daily_ok = opt_daily <= daily_limit
        print(f"    Max DD: ${opt_dd:,.0f} <= ${dd_limit:,.0f} {'✅' if dd_ok else '❌'}")
        print(f"    Daily Loss: ${opt_daily:,.0f} <= ${daily_limit:,.0f} {'✅' if daily_ok else '❌'}")

        if dd_ok and daily_ok:
            print(f"\n✅ Portfolio manifest validation passed")
            return True
        else:
            print(f"\n❌ Constraints violated!")
            return False

    print(f"\n✅ Portfolio manifest validation passed")
    return True


def main():
    print("="*80)
    print("COMPLETE PIPELINE END-TO-END TEST")
    print("="*80)
    print("\nThis test validates all phases of the automated trading pipeline:")
    print("  Phase 1: Strategy Factory → Database")
    print("  Phase 2: Extract Winners → Manifest")
    print("  Phase 3: ML Meta-Labeling → Validation Data (already created)")
    print("  Phase 4: Portfolio Optimization → Deployment Manifest")
    print()

    test_results = []

    # Phase 1: Create sample database
    print_header("PHASE 1: Creating Sample Strategy Factory Database")
    test_results.append(run_command(
        ['python', 'research/strategy_factory/create_sample_database.py'],
        "Create Strategy Factory database with backtest results"
    ))

    # Validate Phase 1
    db_path = Path('test_data/strategy_factory/results/strategy_factory.db')
    test_results.append(validate_database(db_path))

    # Phase 2: Extract winners
    print_header("PHASE 2: Extracting Top Winners")
    test_results.append(run_command(
        [
            'python', 'research/strategy_factory/extract_winners.py',
            '--db-path', 'test_data/strategy_factory/results/strategy_factory.db',
            '--top-n', '5',
            '--output', 'test_data/winners_manifest.json'
        ],
        "Extract top 5 winners per instrument"
    ))

    # Validate Phase 2
    manifest_path = Path('test_data/winners_manifest.json')
    test_results.append(validate_winners_manifest(manifest_path))

    # Phase 3: ML Validation Data (use pre-created data)
    validation_dir = Path('test_data/ml_meta_labeling/results')
    test_results.append(validate_ml_validation_data(validation_dir))

    # Phase 4: Portfolio Optimization
    print_header("PHASE 4: Running Portfolio Optimization")
    test_results.append(run_command(
        [
            'python', 'research/portfolio_optimization/portfolio_optimizer.py',
            '--validation-dir', 'test_data/ml_meta_labeling/results',
            '--max-drawdown', '6000',
            '--daily-loss-limit', '3000',
            '--output', 'test_data/portfolio_manifest.json'
        ],
        "Optimize portfolio selection"
    ))

    # Validate Phase 4
    portfolio_path = Path('test_data/portfolio_manifest.json')
    test_results.append(validate_portfolio_manifest(portfolio_path))

    # Summary
    print("\n\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    phase_names = [
        "Phase 1: Create Sample Database",
        "Phase 1: Validate Database",
        "Phase 2: Extract Winners",
        "Phase 2: Validate Manifest",
        "Phase 3: Validate ML Data",
        "Phase 4: Portfolio Optimization",
        "Phase 4: Validate Portfolio"
    ]

    print(f"\nResults by Phase:")
    for i, (name, result) in enumerate(zip(phase_names, test_results), 1):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {i}. {name}: {status}")

    print(f"\nOverall:")
    print(f"  Total tests: {len(test_results)}")
    print(f"  Passed: {sum(test_results)}")
    print(f"  Failed: {len(test_results) - sum(test_results)}")

    if all(test_results):
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"\nThe complete pipeline is working correctly:")
        print(f"  ✅ Phase 1: Strategy Factory database generation")
        print(f"  ✅ Phase 2: Winner extraction and manifest creation")
        print(f"  ✅ Phase 3: ML validation data available")
        print(f"  ✅ Phase 4: Portfolio optimization with constraints")
        print(f"\n✨ READY FOR PRODUCTION DEPLOYMENT! ✨")
        return 0
    else:
        print(f"\n⚠️  SOME TESTS FAILED")
        print(f"Please review the errors above and fix before deployment.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
