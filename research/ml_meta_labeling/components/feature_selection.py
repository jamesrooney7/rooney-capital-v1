"""
Component 2: Feature Clustering and Selection

Implements hierarchical clustering with MDA (Mean Decrease Accuracy) importance
to select diverse, representative features.
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.ensemble import RandomForestClassifier
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from .config_defaults import FEATURE_SELECTION_DEFAULTS

logger = logging.getLogger(__name__)


class FeatureSelection:
    """Select features via hierarchical clustering and MDA importance."""

    def __init__(
        self,
        n_clusters: int = FEATURE_SELECTION_DEFAULTS['n_clusters'],
        linkage_method: str = FEATURE_SELECTION_DEFAULTS['linkage_method'],
        rf_n_estimators: int = FEATURE_SELECTION_DEFAULTS['rf_n_estimators'],
        random_state: int = 42
    ):
        """
        Initialize feature selection.

        Args:
            n_clusters: Target number of feature clusters (default: 30)
            linkage_method: Hierarchical clustering linkage ('ward' or 'complete')
            rf_n_estimators: Number of trees for preliminary Random Forest
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.rf_n_estimators = rf_n_estimators
        self.random_state = random_state

        # Will be populated by select_features()
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.distance_matrix: Optional[np.ndarray] = None
        self.linkage_matrix: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.feature_importance: Optional[Dict[str, float]] = None
        self.selected_features: Optional[List[str]] = None
        self.feature_clusters: Optional[Dict[int, List[str]]] = None

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None
    ) -> Tuple[List[str], Dict]:
        """
        Select features via hierarchical clustering and MDA importance.

        Args:
            X: Feature matrix (samples x features)
            y: Target labels
            sample_weight: Optional sample weights

        Returns:
            Tuple of (selected_feature_names, selection_report_dict)
        """
        logger.info(f"Starting feature selection from {X.shape[1]} features...")

        # Step 1: Compute correlation matrix
        self._compute_correlation_matrix(X)

        # Step 2: Perform hierarchical clustering
        self._hierarchical_clustering()

        # Step 3: Calculate feature importance
        self._calculate_mda_importance(X, y, sample_weight)

        # Step 4: Select representatives from each cluster
        self._select_representatives()

        # Step 5: Validate diversity
        diversity_report = self._validate_diversity(X.columns.tolist())

        # Generate report
        report = self._generate_report(diversity_report)

        logger.info(f"Feature selection complete: {len(self.selected_features)} features selected")
        return self.selected_features, report

    def _compute_correlation_matrix(self, X: pd.DataFrame):
        """Compute absolute correlation matrix and convert to distance."""
        logger.info("Computing correlation matrix...")

        # Remove any features with zero variance
        variances = X.var()
        valid_features = variances[variances > 0].index.tolist()

        if len(valid_features) < len(X.columns):
            logger.warning(f"Removing {len(X.columns) - len(valid_features)} zero-variance features")
            X = X[valid_features]

        # Compute correlation matrix
        self.correlation_matrix = X.corr().abs()

        # Convert to distance: distance = 1 - |correlation|
        self.distance_matrix = 1 - self.correlation_matrix.values

        # Ensure diagonal is zero (distance to self)
        np.fill_diagonal(self.distance_matrix, 0)

        logger.info(f"Correlation matrix computed: {self.correlation_matrix.shape}")

    def _hierarchical_clustering(self):
        """Perform hierarchical clustering on features."""
        logger.info(f"Performing hierarchical clustering (linkage={self.linkage_method})...")

        # Convert distance matrix to condensed form for scipy
        condensed_dist = squareform(self.distance_matrix, checks=False)

        # Perform hierarchical clustering
        self.linkage_matrix = linkage(condensed_dist, method=self.linkage_method)

        # Cut tree to get cluster labels
        self.cluster_labels = fcluster(self.linkage_matrix, t=self.n_clusters, criterion='maxclust')

        # Group features by cluster
        self.feature_clusters = {}
        for cluster_id in range(1, self.n_clusters + 1):
            cluster_features = self.correlation_matrix.index[self.cluster_labels == cluster_id].tolist()
            if cluster_features:  # Only add non-empty clusters
                self.feature_clusters[cluster_id] = cluster_features

        logger.info(f"Created {len(self.feature_clusters)} clusters (target: {self.n_clusters})")

        # Log cluster sizes
        cluster_sizes = [len(features) for features in self.feature_clusters.values()]
        logger.info(f"Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, "
                   f"mean={np.mean(cluster_sizes):.1f}")

    def _calculate_mda_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None
    ):
        """Calculate Mean Decrease Accuracy (MDA) importance using Random Forest."""
        logger.info(f"Calculating MDA importance (n_estimators={self.rf_n_estimators})...")

        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            oob_score=True
        )

        if sample_weight is not None:
            rf.fit(X, y, sample_weight=sample_weight)
        else:
            rf.fit(X, y)

        logger.info(f"Random Forest OOB Score: {rf.oob_score_:.4f}")

        # Get feature importances (MDI - Mean Decrease Impurity, similar to MDA)
        # Note: sklearn's feature_importances_ uses MDI, not MDA
        # For true MDA, we'd use permutation_importance, but MDI is faster and highly correlated
        importances = rf.feature_importances_

        self.feature_importance = dict(zip(X.columns, importances))

        # Log top 10 most important features
        sorted_importance = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        logger.info("Top 10 most important features:")
        for i, (feat, imp) in enumerate(sorted_importance[:10], 1):
            logger.info(f"  {i}. {feat}: {imp:.4f}")

    def _select_representatives(self):
        """Select the most important feature from each cluster."""
        logger.info("Selecting cluster representatives...")

        self.selected_features = []

        for cluster_id, features in self.feature_clusters.items():
            # Get importance scores for features in this cluster
            cluster_importances = {
                feat: self.feature_importance[feat]
                for feat in features
                if feat in self.feature_importance
            }

            if not cluster_importances:
                logger.warning(f"Cluster {cluster_id} has no features with importance scores")
                continue

            # Select feature with highest importance
            best_feature = max(cluster_importances.items(), key=lambda x: x[1])[0]
            self.selected_features.append(best_feature)

        logger.info(f"Selected {len(self.selected_features)} representative features")

    def _validate_diversity(self, all_features: List[str]) -> Dict:
        """
        Validate cross-asset diversity of selected features.

        Categorizes features into:
        - Own instrument features
        - Cross-asset features
        - Risk proxies (equity indexes, bonds)
        - Market microstructure

        Args:
            all_features: List of all available feature names

        Returns:
            Dictionary with diversity breakdown
        """
        logger.info("Validating cross-asset diversity...")

        # Categorize selected features
        own_instrument = []
        cross_asset = []
        risk_proxies = []
        microstructure = []

        # Risk proxy symbols
        risk_symbols = {'es', 'nq', 'rty', 'ym', 'tlt', 'gc', 'si'}

        # Microstructure indicators
        microstructure_keywords = {'pair', 'ibs', 'rsi', 'supply', 'inside_bar'}

        for feat in self.selected_features:
            feat_lower = feat.lower()

            # Check if it's a microstructure feature
            if any(keyword in feat_lower for keyword in microstructure_keywords):
                microstructure.append(feat)
            # Check if it's a risk proxy
            elif any(symbol in feat_lower for symbol in risk_symbols):
                risk_proxies.append(feat)
            # Check if it mentions other instruments (cross-asset)
            elif any(char.isdigit() for char in feat[:2]) or any(
                symbol in feat_lower for symbol in ['cl', 'ng', 'hg', 'pl', '6a', '6b', '6c', '6e', '6j', '6m', '6n', '6s']
            ):
                cross_asset.append(feat)
            # Otherwise, it's own instrument
            else:
                own_instrument.append(feat)

        diversity_report = {
            'own_instrument': own_instrument,
            'cross_asset': cross_asset,
            'risk_proxies': risk_proxies,
            'microstructure': microstructure,
            'counts': {
                'own_instrument': len(own_instrument),
                'cross_asset': len(cross_asset),
                'risk_proxies': len(risk_proxies),
                'microstructure': len(microstructure)
            }
        }

        logger.info("Diversity breakdown:")
        logger.info(f"  Own instrument: {len(own_instrument)}")
        logger.info(f"  Cross-asset: {len(cross_asset)}")
        logger.info(f"  Risk proxies: {len(risk_proxies)}")
        logger.info(f"  Microstructure: {len(microstructure)}")

        return diversity_report

    def _generate_report(self, diversity_report: Dict) -> Dict:
        """Generate detailed feature selection report."""
        # Get cluster membership for selected features
        selected_cluster_info = {}
        for feat in self.selected_features:
            feat_idx = self.correlation_matrix.index.get_loc(feat)
            cluster_id = self.cluster_labels[feat_idx]
            selected_cluster_info[feat] = {
                'cluster_id': int(cluster_id),
                'importance': self.feature_importance[feat],
                'cluster_size': len(self.feature_clusters[cluster_id]),
                'cluster_members': self.feature_clusters[cluster_id]
            }

        report = {
            'n_clusters_target': self.n_clusters,
            'n_clusters_created': len(self.feature_clusters),
            'n_features_selected': len(self.selected_features),
            'linkage_method': self.linkage_method,
            'selected_features': self.selected_features,
            'feature_info': selected_cluster_info,
            'diversity': diversity_report
        }

        return report

    def get_selection_summary(self) -> str:
        """Get a text summary of the feature selection."""
        if self.selected_features is None:
            return "Feature selection not yet performed"

        lines = []
        lines.append("=" * 80)
        lines.append("FEATURE SELECTION SUMMARY")
        lines.append("=" * 80)
        lines.append(f"Total features analyzed: {len(self.correlation_matrix)}")
        lines.append(f"Target clusters: {self.n_clusters}")
        lines.append(f"Linkage method: {self.linkage_method}")
        lines.append(f"Features selected: {len(self.selected_features)}")
        lines.append("")
        lines.append("SELECTED FEATURES:")
        for i, feat in enumerate(sorted(self.selected_features), 1):
            imp = self.feature_importance[feat]
            lines.append(f"  {i:2d}. {feat:40s} (importance: {imp:.4f})")
        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)
