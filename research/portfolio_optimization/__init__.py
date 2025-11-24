"""
Portfolio Optimization System

Equal-weighted portfolio optimizer with risk constraints.
Selects optimal combination of strategies to maximize Sharpe while respecting:
- Max drawdown ≤ $6,000
- Daily loss limit ≤ $3,000

Usage:
    from research.portfolio_optimization.portfolio_optimizer import PortfolioOptimizer
    from research.portfolio_optimization.risk_manager import PortfolioRiskManager
"""

from .portfolio_optimizer import PortfolioOptimizer
from .risk_manager import PortfolioRiskManager

__all__ = ['PortfolioOptimizer', 'PortfolioRiskManager']
