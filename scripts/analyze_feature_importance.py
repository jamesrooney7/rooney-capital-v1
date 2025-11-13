#!/usr/bin/env python3
"""
Analyze feature importance from trained RF models.
Shows which features are driving predictions and if they make logical sense.
"""

import joblib
import json
from pathlib import Path
import pandas as pd


def analyze_model(symbol):
    """Analyze feature importance for a symbol."""
    model_path = Path('src/models') / f'{symbol}_rf_model.pkl'
    meta_path = Path('src/models') / f'{symbol}_best.json'

    if not model_path.exists():
        return None

    # Load model
    model_data = joblib.load(model_path)
    model = model_data['model']
    features = model_data['features']

    # Load metadata
    with open(meta_path) as f:
        meta = json.load(f)

    # Get feature importances
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=False)

    return {
        'symbol': symbol,
        'feature_importance': feature_importance,
        'threshold': meta['threshold'],
        'params': meta['params'],
    }


def main():
    symbols = ['ES', 'NQ', 'RTY', 'YM']

    print("=" * 100)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 100)
    print()

    for symbol in symbols:
        result = analyze_model(symbol)
        if not result:
            print(f"❌ {symbol}: Model not found")
            continue

        print("=" * 100)
        print(f"{symbol} - TOP 20 MOST IMPORTANT FEATURES")
        print("=" * 100)
        print()

        fi = result['feature_importance'].head(20)

        # Calculate cumulative importance
        fi['cumulative'] = fi['importance'].cumsum()

        print(f"{'Rank':<6} {'Feature':<40} {'Importance':<12} {'Cumulative':<12}")
        print("-" * 100)

        for idx, (i, row) in enumerate(fi.iterrows(), 1):
            print(f"{idx:<6} {row['feature']:<40} {row['importance']:<12.4f} {row['cumulative']:<12.2%}")

        print()

        # Show how many features needed for 80% importance
        fi_full = result['feature_importance']
        cumsum = fi_full['importance'].cumsum()
        n_features_80 = (cumsum >= 0.80).argmax() + 1

        print(f"📊 Feature Statistics:")
        print(f"  Total features: {len(fi_full)}")
        print(f"  Top feature importance: {fi_full.iloc[0]['importance']:.2%}")
        print(f"  Top 5 importance: {fi_full.head(5)['importance'].sum():.2%}")
        print(f"  Top 10 importance: {fi_full.head(10)['importance'].sum():.2%}")
        print(f"  Features for 80% importance: {n_features_80}")
        print()

        # Check for suspicious patterns
        print(f"🔍 Validation Checks:")

        # Check if any single feature dominates (>30% = suspicious)
        top_feature_pct = fi_full.iloc[0]['importance']
        if top_feature_pct > 0.30:
            print(f"  ⚠️  WARNING: Top feature has {top_feature_pct:.1%} importance (>30% may indicate leakage)")
        else:
            print(f"  ✅ Top feature importance reasonable: {top_feature_pct:.1%}")

        # Check feature diversity
        if n_features_80 < 5:
            print(f"  ⚠️  WARNING: Only {n_features_80} features needed for 80% importance (low diversity)")
        else:
            print(f"  ✅ Good feature diversity: {n_features_80} features for 80%")

        # Identify feature categories from top 10
        top_10_names = fi_full.head(10)['feature'].tolist()

        categories = {
            'reference_returns': [f for f in top_10_names if any(x in f.lower() for x in ['_return', 'hourly_return', 'daily_return'])],
            'z_scores': [f for f in top_10_names if 'z_score' in f.lower()],
            'rsi_ibs': [f for f in top_10_names if any(x in f.lower() for x in ['rsi', 'ibs'])],
            'volatility': [f for f in top_10_names if any(x in f.lower() for x in ['atr', 'vol'])],
            'price_patterns': [f for f in top_10_names if any(x in f.lower() for x in ['bb_', 'ema', 'sma'])],
        }

        print(f"\n  📋 Top 10 Feature Categories:")
        for cat, feats in categories.items():
            if feats:
                print(f"    {cat}: {len(feats)} features - {', '.join(feats[:3])}")

        print()

    # Cross-symbol comparison
    print("=" * 100)
    print("CROSS-SYMBOL COMPARISON - Top 5 Features")
    print("=" * 100)
    print()

    all_results = {s: analyze_model(s) for s in symbols}
    all_results = {k: v for k, v in all_results.items() if v}

    for symbol, result in all_results.items():
        top_5 = result['feature_importance'].head(5)
        print(f"{symbol}:")
        for i, row in top_5.iterrows():
            print(f"  {row['feature']:<40} {row['importance']:.4f}")
        print()


if __name__ == '__main__':
    main()
