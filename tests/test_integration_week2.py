"""
Integration tests for Week 2: IBS Strategy Migration

Tests:
1. Strategy registration in factory
2. Strategy loading from configuration
3. Strategy initialization
4. BaseStrategy interface compliance
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


def test_strategy_registration():
    """Test that IBS strategy is registered in the factory."""
    logger.info("="*80)
    logger.info("TEST 1: Strategy Registration")
    logger.info("="*80)

    try:
        from src.strategy.strategy_factory import get_registered_strategies

        strategies = get_registered_strategies()
        logger.info(f"Registered strategies: {list(strategies.keys())}")

        if 'ibs' not in strategies:
            logger.error("❌ IBS strategy NOT registered")
            return False

        logger.info(f"✅ IBS strategy registered: {strategies['ibs']}")
        return True

    except Exception as e:
        logger.error(f"❌ Strategy registration test failed: {e}", exc_info=True)
        return False


def test_strategy_loading():
    """Test loading IBS strategy from factory."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Strategy Loading")
    logger.info("="*80)

    try:
        from src.strategy.strategy_factory import load_strategy

        strategy_class = load_strategy("ibs")
        logger.info(f"✅ Loaded strategy class: {strategy_class}")
        logger.info(f"   Strategy name: {strategy_class.__name__}")
        logger.info(f"   Base classes: {[c.__name__ for c in strategy_class.__bases__]}")

        return True

    except Exception as e:
        logger.error(f"❌ Strategy loading test failed: {e}", exc_info=True)
        return False


def test_base_strategy_compliance():
    """Test that IBS strategy properly implements BaseStrategy interface."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: BaseStrategy Interface Compliance")
    logger.info("="*80)

    try:
        from src.strategy.ibs_strategy import IbsStrategy
        from src.strategy.base_strategy import BaseStrategy
        from abc import ABC

        # Check inheritance
        if not issubclass(IbsStrategy, BaseStrategy):
            logger.error("❌ IbsStrategy does not inherit from BaseStrategy")
            return False
        logger.info("✅ IbsStrategy extends BaseStrategy")

        # Check required methods exist
        required_methods = [
            'should_enter_long',
            'should_enter_short',
            'should_exit',
            'get_features_snapshot'
        ]

        for method in required_methods:
            if not hasattr(IbsStrategy, method):
                logger.error(f"❌ Missing required method: {method}")
                return False
            logger.info(f"✅ Has method: {method}")

        # Check methods are callable
        for method in required_methods:
            method_obj = getattr(IbsStrategy, method)
            if not callable(method_obj):
                logger.error(f"❌ Method not callable: {method}")
                return False

        logger.info("✅ All required methods are present and callable")
        return True

    except Exception as e:
        logger.error(f"❌ BaseStrategy compliance test failed: {e}", exc_info=True)
        return False


def test_config_loading():
    """Test loading configuration file."""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Configuration Loading")
    logger.info("="*80)

    try:
        from src.config.config_loader import load_config

        config_path = Path(__file__).parent.parent / "config.test.yml"
        if not config_path.exists():
            logger.error(f"❌ Test config not found: {config_path}")
            return False

        config = load_config(str(config_path))
        logger.info(f"✅ Loaded config from: {config_path}")

        # Check IBS strategy config
        if 'ibs' not in config.strategies:
            logger.error("❌ IBS strategy not in config")
            return False

        ibs_config = config.strategies['ibs']
        logger.info(f"✅ IBS strategy config loaded")
        logger.info(f"   Enabled: {ibs_config.enabled}")
        logger.info(f"   Instruments: {ibs_config.instruments}")
        logger.info(f"   Starting cash: ${ibs_config.starting_cash:,.0f}")
        logger.info(f"   Max positions: {ibs_config.max_positions}")

        return True

    except Exception as e:
        logger.error(f"❌ Config loading test failed: {e}", exc_info=True)
        return False


def test_strategy_params_creation():
    """Test creating strategy parameters from config."""
    logger.info("\n" + "="*80)
    logger.info("TEST 5: Strategy Parameters Creation")
    logger.info("="*80)

    try:
        from src.config.config_loader import load_config
        from src.strategy.strategy_factory import create_strategy_config

        config_path = Path(__file__).parent.parent / "config.test.yml"
        config = load_config(str(config_path))

        ibs_config = config.strategies['ibs']

        # Create strategy params
        strategy_params = create_strategy_config(
            strategy_name='ibs',
            symbol='ES',
            config=ibs_config.__dict__,
            portfolio_coordinator=None,
            ml_model=None,
            ml_features=None,
            ml_threshold=ibs_config.strategy_params.get('ml_threshold', 0.65)
        )

        logger.info(f"✅ Created strategy params:")
        for key, value in sorted(strategy_params.items()):
            if value is not None and value != '':
                logger.info(f"   {key}: {value}")

        # Verify essential params
        essential_params = ['strategy_name', 'symbol', 'size']
        for param in essential_params:
            if param not in strategy_params:
                logger.error(f"❌ Missing essential param: {param}")
                return False

        logger.info("✅ All essential parameters present")
        return True

    except Exception as e:
        logger.error(f"❌ Strategy params creation test failed: {e}", exc_info=True)
        return False


def run_all_tests():
    """Run all integration tests."""
    logger.info("\n")
    logger.info("#" * 80)
    logger.info("# WEEK 2 INTEGRATION TESTS: IBS Strategy Migration")
    logger.info("#" * 80)
    logger.info("\n")

    tests = [
        ("Strategy Registration", test_strategy_registration),
        ("Strategy Loading", test_strategy_loading),
        ("BaseStrategy Compliance", test_base_strategy_compliance),
        ("Configuration Loading", test_config_loading),
        ("Strategy Params Creation", test_strategy_params_creation),
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
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
