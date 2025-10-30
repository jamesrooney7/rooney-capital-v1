"""Performance metrics for trading strategy evaluation.

This module implements advanced statistical metrics for evaluating trading strategies,
with a focus on multiple testing corrections and statistical significance.

Key metric:
- Deflated Sharpe Ratio (DSR): Corrects for multiple testing bias in strategy optimization

References:
- Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting, and Non-Normality." Journal of Portfolio Management.
"""

import logging
import math
from typing import Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def calculate_deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int = 90,
    rho: float = 0.5,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    n_observations: Optional[int] = None,
) -> float:
    """Calculate Deflated Sharpe Ratio (DSR) to correct for multiple testing bias.

    The DSR adjusts an observed Sharpe ratio for:
    1. Multiple testing: Testing N strategies inflates expected max Sharpe by chance
    2. Strategy correlation: Correlated strategies reduce effective number of independent tests
    3. Non-normality: Skewness and kurtosis affect Sharpe ratio distribution

    Formula:
        DSR = (observed_SR - E[max_SR]) / σ[max_SR]

    Where:
        E[max_SR] = √(2 × log(N_eff))  [expected maximum Sharpe under null hypothesis]
        N_eff = N / (1 + (N-1) × ρ)  [effective number of independent trials]
        σ[max_SR] = variance adjustment for non-normality

    Interpretation:
        DSR > 0.95: Statistically significant (95% confidence)
        DSR > 0.0: Better than expected by chance
        DSR < 0.0: Worse than expected by chance (likely overfit)

    Args:
        observed_sharpe: Observed Sharpe ratio from strategy
        n_trials: Number of strategies tested (default: 90, reduced from 420 for meta-labeling)
        rho: Average pairwise correlation between strategies (default: 0.5)
        skewness: Return distribution skewness (default: 0.0 for normal)
        kurtosis: Return distribution kurtosis (default: 3.0 for normal)
        n_observations: Number of returns used to calculate Sharpe (optional, for variance adjustment)

    Returns:
        Deflated Sharpe Ratio (z-score)
    """
    # Step 1: Calculate effective number of independent trials
    n_eff = n_trials / (1.0 + (n_trials - 1.0) * rho)

    # Step 2: Expected maximum Sharpe ratio under null hypothesis
    # Derived from extreme value theory: max of N independent standard normals
    expected_max_sr = math.sqrt(2.0 * math.log(n_eff))

    # Step 3: Variance of maximum Sharpe ratio
    # Standard formula from extreme value theory
    gamma = 0.5772156649  # Euler-Mascheroni constant
    sigma_max_sr = (1.0 / expected_max_sr) * (1.0 - gamma * expected_max_sr + (math.pi**2 / 6.0) * (expected_max_sr**2))

    # Step 4: Adjust for non-normality (skewness and excess kurtosis)
    # Bailey & López de Prado correction for non-normal returns
    if n_observations is not None and n_observations > 0:
        excess_kurtosis = kurtosis - 3.0
        non_normality_adjustment = (
            skewness / math.sqrt(6.0 * n_observations)
            + (kurtosis - 3.0) / math.sqrt(24.0 * n_observations)
        )
        sigma_max_sr = sigma_max_sr * (1.0 + non_normality_adjustment)

    # Step 5: Calculate deflated Sharpe ratio (z-score)
    dsr = (observed_sharpe - expected_max_sr) / sigma_max_sr

    logger.debug(
        f"DSR Calculation: observed_SR={observed_sharpe:.3f}, N_trials={n_trials}, "
        f"N_eff={n_eff:.1f}, E[max_SR]={expected_max_sr:.3f}, σ[max_SR]={sigma_max_sr:.3f}, "
        f"DSR={dsr:.3f}"
    )

    return float(dsr)


def get_required_sharpe(
    n_trials: int = 90,
    rho: float = 0.5,
    confidence: float = 0.95,
) -> float:
    """Calculate minimum Sharpe ratio required for statistical significance.

    Args:
        n_trials: Number of strategies tested (default: 90)
        rho: Average pairwise correlation (default: 0.5)
        confidence: Confidence level (default: 0.95 for 95% confidence)

    Returns:
        Minimum Sharpe ratio required for significance
    """
    # Effective number of trials
    n_eff = n_trials / (1.0 + (n_trials - 1.0) * rho)

    # Expected maximum Sharpe under null
    expected_max_sr = math.sqrt(2.0 * math.log(n_eff))

    # Variance of maximum Sharpe
    gamma = 0.5772156649
    sigma_max_sr = (1.0 / expected_max_sr) * (1.0 - gamma * expected_max_sr + (math.pi**2 / 6.0) * (expected_max_sr**2))

    # Z-score for desired confidence level
    z_score = stats.norm.ppf(confidence)

    # Required Sharpe: E[max_SR] + z * σ[max_SR]
    required_sharpe = expected_max_sr + z_score * sigma_max_sr

    return float(required_sharpe)


def validate_sharpe_significance(
    observed_sharpe: float,
    n_trials: int = 90,
    rho: float = 0.5,
    confidence: float = 0.95,
) -> Tuple[bool, str]:
    """Check if observed Sharpe ratio is statistically significant.

    Args:
        observed_sharpe: Observed Sharpe ratio
        n_trials: Number of strategies tested (default: 90)
        rho: Average pairwise correlation (default: 0.5)
        confidence: Confidence level (default: 0.95)

    Returns:
        Tuple of (is_significant, explanation_message)
    """
    # Calculate DSR
    dsr = calculate_deflated_sharpe_ratio(observed_sharpe, n_trials, rho)

    # Calculate required Sharpe
    required_sr = get_required_sharpe(n_trials, rho, confidence)

    # Z-score threshold for confidence level
    z_threshold = stats.norm.ppf(confidence)

    # Check significance
    is_significant = dsr >= z_threshold

    # Generate explanation
    if is_significant:
        message = (
            f"✅ SIGNIFICANT: Observed Sharpe={observed_sharpe:.3f} exceeds required={required_sr:.3f} "
            f"(DSR={dsr:.2f} >= {z_threshold:.2f})\n"
            f"   This result is statistically significant at {confidence*100:.0f}% confidence "
            f"after correcting for {n_trials} trials."
        )
    else:
        shortfall = required_sr - observed_sharpe
        message = (
            f"❌ NOT SIGNIFICANT: Observed Sharpe={observed_sharpe:.3f} below required={required_sr:.3f} "
            f"(DSR={dsr:.2f} < {z_threshold:.2f})\n"
            f"   Shortfall: {shortfall:.3f} Sharpe points\n"
            f"   This could be due to overfitting or luck from testing {n_trials} strategies."
        )

    return is_significant, message


def print_dsr_analysis(
    observed_sharpe: float,
    n_trials: int = 90,
    rho: float = 0.5,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    n_observations: Optional[int] = None,
) -> None:
    """Print comprehensive DSR analysis for a strategy.

    Args:
        observed_sharpe: Observed Sharpe ratio
        n_trials: Number of strategies tested (default: 90)
        rho: Average pairwise correlation (default: 0.5)
        skewness: Return distribution skewness (default: 0.0)
        kurtosis: Return distribution kurtosis (default: 3.0)
        n_observations: Number of returns (optional)
    """
    print("\n" + "=" * 70)
    print("DEFLATED SHARPE RATIO (DSR) ANALYSIS")
    print("=" * 70)

    # Calculate DSR
    dsr = calculate_deflated_sharpe_ratio(
        observed_sharpe, n_trials, rho, skewness, kurtosis, n_observations
    )

    # Calculate required Sharpe for different confidence levels
    required_95 = get_required_sharpe(n_trials, rho, 0.95)
    required_99 = get_required_sharpe(n_trials, rho, 0.99)

    # Effective number of trials
    n_eff = n_trials / (1.0 + (n_trials - 1.0) * rho)

    # Expected max Sharpe under null
    expected_max_sr = math.sqrt(2.0 * math.log(n_eff))

    print(f"\nInput Parameters:")
    print(f"  Observed Sharpe Ratio: {observed_sharpe:.3f}")
    print(f"  Number of trials tested: {n_trials}")
    print(f"  Average correlation (ρ): {rho:.2f}")
    print(f"  Effective independent trials: {n_eff:.1f}")

    print(f"\nMultiple Testing Correction:")
    print(f"  Expected max SR (under null): {expected_max_sr:.3f}")
    print(f"  Deflated Sharpe Ratio (DSR): {dsr:.3f}")

    print(f"\nStatistical Significance:")
    print(f"  Required Sharpe (95% confidence): {required_95:.3f}")
    print(f"  Required Sharpe (99% confidence): {required_99:.3f}")

    # Check significance
    if dsr >= 1.645:  # 95% confidence (one-tailed)
        status = "✅ SIGNIFICANT (95%)"
    elif dsr >= 1.282:  # 90% confidence
        status = "⚠️  MARGINAL (90%)"
    elif dsr >= 0.0:
        status = "❓ WEAK (better than chance, but not significant)"
    else:
        status = "❌ NOT SIGNIFICANT (worse than expected by chance)"

    print(f"  Status: {status}")

    print(f"\nInterpretation:")
    if observed_sharpe >= required_95:
        improvement = ((observed_sharpe - expected_max_sr) / expected_max_sr) * 100
        print(f"  Your strategy's Sharpe ({observed_sharpe:.3f}) exceeds the 95% threshold ({required_95:.3f}).")
        print(f"  This is {improvement:.1f}% better than expected by random chance.")
        print(f"  Conclusion: Strategy shows genuine edge, not likely due to overfitting.")
    else:
        shortfall = required_95 - observed_sharpe
        print(f"  Your strategy's Sharpe ({observed_sharpe:.3f}) is below the 95% threshold ({required_95:.3f}).")
        print(f"  Shortfall: {shortfall:.3f} Sharpe points")
        print(f"  Conclusion: Result could be explained by luck or overfitting from testing {n_trials} strategies.")

    print("=" * 70 + "\n")


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Strategy with Sharpe 1.5 after testing 90 models
    print("Example 1: Sharpe=1.5, N=90 trials (meta-labeling optimized)")
    print_dsr_analysis(observed_sharpe=1.5, n_trials=90, rho=0.5)

    # Example 2: Same Sharpe but tested 420 models (old approach)
    print("\nExample 2: Sharpe=1.5, N=420 trials (old approach)")
    print_dsr_analysis(observed_sharpe=1.5, n_trials=420, rho=0.5)

    # Example 3: Higher Sharpe with 90 trials
    print("\nExample 3: Sharpe=2.0, N=90 trials")
    print_dsr_analysis(observed_sharpe=2.0, n_trials=90, rho=0.5)

    # Demonstrate required Sharpe reduction from N=420 to N=90
    print("\n" + "=" * 70)
    print("IMPACT OF TRIAL REDUCTION (420 → 90)")
    print("=" * 70)
    required_420 = get_required_sharpe(n_trials=420, rho=0.5, confidence=0.95)
    required_90 = get_required_sharpe(n_trials=90, rho=0.5, confidence=0.95)
    reduction = ((required_90 - required_420) / required_420) * 100

    print(f"\nRequired Sharpe for 95% confidence:")
    print(f"  With N=420 trials: {required_420:.3f}")
    print(f"  With N=90 trials:  {required_90:.3f}")
    print(f"  Reduction: {abs(reduction):.1f}% easier to achieve significance")
    print(f"\nConclusion: Reducing trials makes it much easier to demonstrate genuine edge!")
    print("=" * 70 + "\n")
