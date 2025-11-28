"""
Strategy Factory - Systematic Strategy Research Pipeline

This package implements a gold-standard methodology for discovering,
testing, and validating trading strategies that feed into the ML pipeline.

Three-Phase Approach:
- Phase 1: Raw strategy screening (ES only, 2010-2024)
- Phase 2: Multi-symbol validation (all symbols)
- Phase 3: ML pipeline integration (extract features → train models)

Key Components:
- strategies/: Strategy implementations (BaseStrategy + 10 Tier 1 strategies)
- engine/: Backtesting, optimization, filtering, statistical testing
- database/: SQLite results storage and querying
- reporting/: Report generation, visualizations, charts
- integration/: ML pipeline integration (Phase 3)
"""

__version__ = "1.0.0"
__author__ = "Rooney Capital"
