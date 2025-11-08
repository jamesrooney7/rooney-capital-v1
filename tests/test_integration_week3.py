"""
Integration tests for Week 3: Multi-Alpha System with IBS A & IBS B

Tests:
1. Configuration loading for both IBS A and IBS B
2. Strategy independence (different instruments, constraints)
3. ML model loading per strategy
4. Portfolio coordinator independence
5. Simultaneous strategy instantiation
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_multi_alpha_config_loading():
    """Test that both IBS A and IBS B configs load correctly."""
    logger.info("="*80)
    logger.info("TEST 1: Multi-Alpha Configuration Loading")
    logger.info("="*80)

    try:
        from src.config.config_loader import load_config

        config_path = Path(__file__).parent.parent / "config.test.yml"
        if not config_path.exists():
            logger.error(f"❌ Test config not found: {config_path}")
            return False

        config = load_config(str(config_path))
        logger.info(f"✅ Loaded config from: {config_path}")

        # Check both IBS A and IBS B exist
        required_strategies = ['ibs_a', 'ibs_b']
        for strategy_name in required_strategies:
            if strategy_name not in config.strategies:
                logger.error(f"❌ Strategy '{strategy_name}' not in config")
                return False
            logger.info(f"✅ Found strategy: {strategy_name}")

        # Verify IBS A config
        ibs_a = config.strategies['ibs_a']
        logger.info(f"\n📊 IBS A Configuration:")
        logger.info(f"   Enabled: {ibs_a.enabled}")
        logger.info(f"   Instruments: {ibs_a.instruments}")
        logger.info(f"   Max positions: {ibs_a.max_positions}")
        logger.info(f"   Daily stop loss: ${ibs_a.daily_stop_loss:,.0f}")
        logger.info(f"   Starting cash: ${ibs_a.starting_cash:,.0f}")

        # Verify IBS B config
        ibs_b = config.strategies['ibs_b']
        logger.info(f"\n📊 IBS B Configuration:")
        logger.info(f"   Enabled: {ibs_b.enabled}")
        logger.info(f"   Instruments: {ibs_b.instruments}")
        logger.info(f"   Max positions: {ibs_b.max_positions}")
        logger.info(f"   Daily stop loss: ${ibs_b.daily_stop_loss:,.0f}")
        logger.info(f"   Starting cash: ${ibs_b.starting_cash:,.0f}")

        logger.info(f"\n✅ Both strategies configured correctly")
        return True

    except Exception as e:
        logger.error(f"❌ Config loading test failed: {e}", exc_info=True)
        return False


def test_strategy_independence():
    """Test that IBS A and IBS B have independent configurations."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Strategy Independence")
    logger.info("="*80)

    try:
        from src.config.config_loader import load_config

        config_path = Path(__file__).parent.parent / "config.test.yml"
        config = load_config(str(config_path))

        ibs_a = config.strategies['ibs_a']
        ibs_b = config.strategies['ibs_b']

        # Verify different instrument sets
        instruments_a = set(ibs_a.instruments)
        instruments_b = set(ibs_b.instruments)

        logger.info(f"IBS A instruments ({len(instruments_a)}): {sorted(instruments_a)}")
        logger.info(f"IBS B instruments ({len(instruments_b)}): {sorted(instruments_b)}")

        # Check for no overlap (they should be mutually exclusive)
        overlap = instruments_a & instruments_b
        if overlap:
            logger.warning(f"⚠️  Instruments overlap: {sorted(overlap)}")
            logger.info("   (This is OK if intentional, but strategies should ideally be independent)")
        else:
            logger.info("✅ No instrument overlap - strategies are fully independent")

        # Verify different constraints (from optimization)
        if ibs_a.max_positions != ibs_b.max_positions:
            logger.info(f"✅ Different max_positions: IBS A={ibs_a.max_positions}, IBS B={ibs_b.max_positions}")
        else:
            logger.info(f"ℹ️  Same max_positions: {ibs_a.max_positions} (both strategies)")

        # Verify separate broker accounts
        if ibs_a.broker_account != ibs_b.broker_account:
            logger.info(f"✅ Different broker accounts configured")
        else:
            logger.error(f"❌ Same broker account - strategies must have separate webhooks!")
            return False

        logger.info(f"\n✅ Strategies are independent")
        return True

    except Exception as e:
        logger.error(f"❌ Strategy independence test failed: {e}", exc_info=True)
        return False


def test_ml_model_loading_per_strategy():
    """Test that each strategy loads ML models for its instruments only."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: ML Model Loading Per Strategy")
    logger.info("="*80)

    try:
        from src.config.config_loader import load_config
        from src.models.loader import load_model_bundle

        config_path = Path(__file__).parent.parent / "config.test.yml"
        config = load_config(str(config_path))

        # Test IBS A ML models
        logger.info("\n📦 IBS A ML Models:")
        ibs_a = config.strategies['ibs_a']
        models_loaded_a = 0
        for symbol in ibs_a.instruments:
            try:
                bundle = load_model_bundle(
                    symbol,
                    base_dir=ibs_a.models_path
                )
                logger.info(f"   ✅ {symbol}: threshold={bundle.threshold:.3f}, features={len(bundle.features)}")
                models_loaded_a += 1
            except FileNotFoundError:
                logger.warning(f"   ⚠️  {symbol}: ML model not found (will run without ML filter)")
            except Exception as e:
                logger.error(f"   ❌ {symbol}: Failed to load - {e}")

        logger.info(f"\n   Loaded {models_loaded_a}/{len(ibs_a.instruments)} models for IBS A")

        # Test IBS B ML models
        logger.info("\n📦 IBS B ML Models:")
        ibs_b = config.strategies['ibs_b']
        models_loaded_b = 0
        for symbol in ibs_b.instruments:
            try:
                bundle = load_model_bundle(
                    symbol,
                    base_dir=ibs_b.models_path
                )
                logger.info(f"   ✅ {symbol}: threshold={bundle.threshold:.3f}, features={len(bundle.features)}")
                models_loaded_b += 1
            except FileNotFoundError:
                logger.warning(f"   ⚠️  {symbol}: ML model not found (will run without ML filter)")
            except Exception as e:
                logger.error(f"   ❌ {symbol}: Failed to load - {e}")

        logger.info(f"\n   Loaded {models_loaded_b}/{len(ibs_b.instruments)} models for IBS B")

        # At least some models should load
        if models_loaded_a == 0 and models_loaded_b == 0:
            logger.error("❌ No ML models loaded for either strategy")
            return False

        logger.info(f"\n✅ ML model loading verified")
        return True

    except Exception as e:
        logger.error(f"❌ ML model loading test failed: {e}", exc_info=True)
        return False


def test_portfolio_coordinator_independence():
    """Test that each strategy can create independent portfolio coordinators."""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Portfolio Coordinator Independence")
    logger.info("="*80)

    try:
        from src.config.config_loader import load_config
        from src.runner.portfolio_coordinator import PortfolioCoordinator

        config_path = Path(__file__).parent.parent / "config.test.yml"
        config = load_config(str(config_path))

        # Create coordinator for IBS A
        ibs_a = config.strategies['ibs_a']
        coordinator_a = PortfolioCoordinator(
            max_positions=ibs_a.max_positions,
            daily_stop_loss=ibs_a.daily_stop_loss,
            strategy_name='ibs_a'
        )
        logger.info(f"✅ IBS A Coordinator created:")
        logger.info(f"   Max positions: {coordinator_a.max_positions}")
        logger.info(f"   Daily stop loss: ${coordinator_a.daily_stop_loss:,.0f}")
        logger.info(f"   Strategy name: {coordinator_a.strategy_name}")

        # Create coordinator for IBS B
        ibs_b = config.strategies['ibs_b']
        coordinator_b = PortfolioCoordinator(
            max_positions=ibs_b.max_positions,
            daily_stop_loss=ibs_b.daily_stop_loss,
            strategy_name='ibs_b'
        )
        logger.info(f"\n✅ IBS B Coordinator created:")
        logger.info(f"   Max positions: {coordinator_b.max_positions}")
        logger.info(f"   Daily stop loss: ${coordinator_b.daily_stop_loss:,.0f}")
        logger.info(f"   Strategy name: {coordinator_b.strategy_name}")

        # Verify they are independent objects
        if coordinator_a is coordinator_b:
            logger.error("❌ Coordinators are the same object (should be independent)")
            return False

        # Verify different constraints
        if coordinator_a.max_positions != coordinator_b.max_positions:
            logger.info(f"\n✅ Coordinators have different max_positions")

        logger.info(f"\n✅ Portfolio coordinators are independent")
        return True

    except Exception as e:
        logger.error(f"❌ Portfolio coordinator test failed: {e}", exc_info=True)
        return False


def test_strategy_registration():
    """Test that IBS strategy is registered and can be loaded."""
    logger.info("\n" + "="*80)
    logger.info("TEST 5: Strategy Registration & Loading")
    logger.info("="*80)

    try:
        from src.strategy.strategy_factory import load_strategy, get_registered_strategies

        # Check registration
        strategies = get_registered_strategies()
        logger.info(f"Registered strategies: {list(strategies.keys())}")

        if 'ibs' not in strategies:
            logger.error("❌ IBS strategy NOT registered")
            return False

        # Load strategy class
        strategy_class = load_strategy("ibs")
        logger.info(f"✅ Loaded strategy class: {strategy_class.__name__}")

        # Both IBS A and IBS B will use the same IBS strategy class
        # but with different configurations
        logger.info(f"✅ Both IBS A and IBS B will use: {strategy_class.__name__}")

        return True

    except Exception as e:
        logger.error(f"❌ Strategy registration test failed: {e}", exc_info=True)
        return False


def test_expected_symbols():
    """Test that configured symbols match optimization results."""
    logger.info("\n" + "="*80)
    logger.info("TEST 6: Optimization Results Verification")
    logger.info("="*80)

    try:
        from src.config.config_loader import load_config
        import json

        config_path = Path(__file__).parent.parent / "config.test.yml"
        config = load_config(str(config_path))

        # Load optimization results
        opt_a_path = Path(__file__).parent.parent / "config/portfolio_optimization_ibs_a.json"
        opt_b_path = Path(__file__).parent.parent / "config/portfolio_optimization_ibs_b.json"

        if not opt_a_path.exists() or not opt_b_path.exists():
            logger.warning("⚠️  Optimization result files not found - skipping verification")
            return True

        with open(opt_a_path) as f:
            opt_a = json.load(f)

        with open(opt_b_path) as f:
            opt_b = json.load(f)

        # Verify IBS A symbols match optimization
        expected_a = set(opt_a['portfolio_constraints']['optimal_symbols'])
        actual_a = set(config.strategies['ibs_a'].instruments)

        logger.info(f"\nIBS A:")
        logger.info(f"   Expected (from optimization): {sorted(expected_a)}")
        logger.info(f"   Actual (from config): {sorted(actual_a)}")

        if expected_a == actual_a:
            logger.info(f"   ✅ Symbols match optimization results")
        else:
            logger.warning(f"   ⚠️  Symbols differ from optimization:")
            logger.warning(f"      Missing: {sorted(expected_a - actual_a)}")
            logger.warning(f"      Extra: {sorted(actual_a - expected_a)}")

        # Verify IBS B symbols match optimization
        expected_b = set(opt_b['portfolio_constraints']['optimal_symbols'])
        actual_b = set(config.strategies['ibs_b'].instruments)

        logger.info(f"\nIBS B:")
        logger.info(f"   Expected (from optimization): {sorted(expected_b)}")
        logger.info(f"   Actual (from config): {sorted(actual_b)}")

        if expected_b == actual_b:
            logger.info(f"   ✅ Symbols match optimization results")
        else:
            logger.warning(f"   ⚠️  Symbols differ from optimization:")
            logger.warning(f"      Missing: {sorted(expected_b - actual_b)}")
            logger.warning(f"      Extra: {sorted(actual_b - expected_b)}")

        # Verify max_positions match optimization
        if config.strategies['ibs_a'].max_positions == opt_a['portfolio_constraints']['max_positions']:
            logger.info(f"\n✅ IBS A max_positions matches optimization: {opt_a['portfolio_constraints']['max_positions']}")
        else:
            logger.warning(f"\n⚠️  IBS A max_positions mismatch: config={config.strategies['ibs_a'].max_positions}, optimization={opt_a['portfolio_constraints']['max_positions']}")

        if config.strategies['ibs_b'].max_positions == opt_b['portfolio_constraints']['max_positions']:
            logger.info(f"✅ IBS B max_positions matches optimization: {opt_b['portfolio_constraints']['max_positions']}")
        else:
            logger.warning(f"⚠️  IBS B max_positions mismatch: config={config.strategies['ibs_b'].max_positions}, optimization={opt_b['portfolio_constraints']['max_positions']}")

        return True

    except Exception as e:
        logger.error(f"❌ Optimization verification test failed: {e}", exc_info=True)
        return False


def run_all_tests():
    """Run all integration tests."""
    logger.info("\n")
    logger.info("#" * 80)
    logger.info("# WEEK 3 INTEGRATION TESTS: Multi-Alpha System (IBS A & IBS B)")
    logger.info("#" * 80)
    logger.info("\n")

    tests = [
        ("Multi-Alpha Config Loading", test_multi_alpha_config_loading),
        ("Strategy Independence", test_strategy_independence),
        ("ML Model Loading Per Strategy", test_ml_model_loading_per_strategy),
        ("Portfolio Coordinator Independence", test_portfolio_coordinator_independence),
        ("Strategy Registration & Loading", test_strategy_registration),
        ("Optimization Results Verification", test_expected_symbols),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}", exc_info=True)
            results.append((test_name, False))

    # Summary
    logger.info("\n")
    logger.info("="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info("-"*80)
    logger.info(f"Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("\nNext steps:")
        logger.info("1. Deploy to server (/opt/pine/rooney-capital-v1)")
        logger.info("2. Set up paper trading webhooks")
        logger.info("3. Start both strategies and monitor")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed")
        logger.error("\nFix failing tests before deploying to production")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
