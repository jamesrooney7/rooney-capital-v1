"""
Component 6: Walk-Forward Validation Engine

Implements walk-forward validation with per-window reoptimization and
Walk-Forward Efficiency (WFE) calculation.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib

from .config_defaults import WALK_FORWARD_WINDOWS, HELD_OUT_TEST
from .data_preparation import DataPreparation
from .optuna_optimizer import OptunaOptimizer
from .lightgbm_trainer import LightGBMTrainer

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """Walk-forward validation with reoptimization per window."""

    def __init__(
        self,
        data_prep: DataPreparation,
        selected_features: List[str],
        n_trials_per_window: int = 100,
        cv_folds: int = 5,
        embargo_days: int = 60,
        optimization_metric: str = 'precision',
        precision_threshold: float = 0.60,
        random_state: int = 42,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize walk-forward validator.

        Args:
            data_prep: DataPreparation instance with loaded data
            selected_features: List of selected feature names
            n_trials_per_window: Optuna trials per walk-forward window
            cv_folds: Number of CV folds within each window
            embargo_days: Embargo period for Purged K-Fold
            optimization_metric: Metric to optimize ('auc', 'f1', 'precision')
            precision_threshold: Threshold for precision metric (default: 0.60)
            random_state: Random seed
            output_dir: Directory to save intermediate results
        """
        self.data_prep = data_prep
        self.selected_features = selected_features
        self.n_trials_per_window = n_trials_per_window
        self.cv_folds = cv_folds
        self.embargo_days = embargo_days
        self.optimization_metric = optimization_metric
        self.precision_threshold = precision_threshold
        self.random_state = random_state
        self.output_dir = output_dir

        # Results storage
        self.window_results: List[Dict] = []
        self.oos_predictions: pd.DataFrame = pd.DataFrame()
        self.hyperparameter_stability: pd.DataFrame = pd.DataFrame()

    def run_walk_forward(self) -> Tuple[List[Dict], pd.DataFrame]:
        """
        Run complete walk-forward validation.

        Returns:
            Tuple of (window_results_list, oos_predictions_dataframe)
        """
        logger.info("=" * 80)
        logger.info("STARTING WALK-FORWARD VALIDATION")
        logger.info("=" * 80)
        logger.info(f"Number of windows: {len(WALK_FORWARD_WINDOWS)}")
        logger.info(f"Trials per window: {self.n_trials_per_window}")
        logger.info(f"Selected features: {len(self.selected_features)}")
        logger.info("")

        all_oos_predictions = []
        all_hyperparameters = []

        for window_idx, window in enumerate(WALK_FORWARD_WINDOWS, 1):
            logger.info(f"{'=' * 80}")
            logger.info(f"WINDOW {window_idx}/{len(WALK_FORWARD_WINDOWS)}: {window['name']}")
            logger.info(f"{'=' * 80}")

            # Run single window
            window_result = self._run_single_window(window, window_idx)

            # Store results
            self.window_results.append(window_result)
            all_oos_predictions.append(window_result['oos_predictions'])
            all_hyperparameters.append({
                'window': window_idx,
                **window_result['best_hyperparameters']
            })

            logger.info("")

        # Concatenate OOS predictions
        self.oos_predictions = pd.concat(all_oos_predictions, ignore_index=True)

        # Analyze hyperparameter stability
        self.hyperparameter_stability = pd.DataFrame(all_hyperparameters)

        # Calculate aggregate metrics
        self._calculate_aggregate_metrics()

        logger.info("=" * 80)
        logger.info("WALK-FORWARD VALIDATION COMPLETE")
        logger.info("=" * 80)

        return self.window_results, self.oos_predictions

    def _run_single_window(self, window: Dict, window_idx: int) -> Dict:
        """
        Run optimization and evaluation for a single walk-forward window.

        Args:
            window: Window configuration dict
            window_idx: Window number

        Returns:
            Dictionary with window results
        """
        logger.info(f"Training: {window['train_start']} to {window['train_end']}")
        logger.info(f"Testing:  {window['test_start']} to {window['test_end']}")

        # Get training data
        train_df = self.data_prep.get_date_split(
            window['train_start'],
            window['train_end']
        )

        # Get test data
        test_df = self.data_prep.get_date_split(
            window['test_start'],
            window['test_end']
        )

        logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

        # Extract features and targets
        X_train, y_train, sw_train = self.data_prep.get_features_and_target(train_df)
        X_test, y_test, sw_test = self.data_prep.get_features_and_target(test_df)

        # Select only the selected features (add Date for CV)
        X_train_selected = X_train[self.selected_features].copy()
        X_train_selected['Date'] = train_df['Date'].values

        X_test_selected = X_test[self.selected_features].copy()

        # Step 1: Hyperparameter optimization on training window
        logger.info(f"Running Optuna optimization ({self.n_trials_per_window} trials)...")
        logger.info(f"Optimization metric: {self.optimization_metric}")
        if self.optimization_metric == 'precision':
            logger.info(f"Precision threshold: {self.precision_threshold}")

        optimizer = OptunaOptimizer(
            n_trials=self.n_trials_per_window,
            n_jobs=-1,
            optimization_metric=self.optimization_metric,
            precision_threshold=self.precision_threshold,
            cv_folds=self.cv_folds,
            embargo_days=self.embargo_days,
            random_state=self.random_state
        )

        best_params, best_score = optimizer.optimize(
            X_train_selected,
            y_train,
            sw_train
        )

        logger.info(f"Best CV {self.optimization_metric.upper()}: {best_score:.4f}")

        # Step 2: Train final model on full training window
        logger.info("Training final model on full training window...")

        trainer = LightGBMTrainer(
            hyperparameters=best_params,
            random_state=self.random_state
        )

        trainer.train(
            X_train_selected.drop(columns=['Date']),
            y_train,
            sw_train
        )

        # Step 3: Generate out-of-sample predictions on test window
        logger.info("Generating OOS predictions on test window...")

        y_pred_proba = trainer.predict_proba(X_test_selected)

        # Create predictions dataframe
        oos_predictions_df = pd.DataFrame({
            'Date': test_df['Date'].values,
            'y_true': y_test.values,
            'y_pred_proba': y_pred_proba,
            'window': window_idx
        })

        # Step 4: Calculate window metrics
        test_metrics = self._calculate_window_metrics(y_test, y_pred_proba, test_df)
        train_metrics = self._calculate_window_metrics(
            y_train,
            trainer.predict_proba(X_train_selected.drop(columns=['Date'])),
            train_df,
            is_training=True
        )

        # Calculate WFE (Walk-Forward Efficiency)
        wfe = self._calculate_wfe(train_metrics, test_metrics)

        # Save window model if output directory specified
        if self.output_dir:
            model_path = self.output_dir / f"window_{window_idx}_model.pkl"
            trainer.save_model(str(model_path))

        # Compile window result
        window_result = {
            'window_idx': window_idx,
            'window_name': window['name'],
            'train_start': window['train_start'],
            'train_end': window['train_end'],
            'test_start': window['test_start'],
            'test_end': window['test_end'],
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'best_hyperparameters': best_params,
            'cv_score': best_score,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'wfe': wfe,
            'oos_predictions': oos_predictions_df
        }

        # Log summary
        logger.info(f"Window {window_idx} Results:")
        logger.info(f"  CV AUC:    {best_score:.4f}")
        logger.info(f"  Test AUC:  {test_metrics['auc']:.4f}")
        logger.info(f"  Test Sharpe: {test_metrics['sharpe']:.3f}")
        logger.info(f"  WFE:       {wfe:.2%}")

        return window_result

    def _calculate_window_metrics(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        df: pd.DataFrame,
        is_training: bool = False
    ) -> Dict:
        """Calculate performance metrics for a window."""
        from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

        # Classification metrics
        y_pred = (y_pred_proba >= 0.5).astype(int)

        metrics = {
            'auc': roc_auc_score(y_true, y_pred_proba),
            'f1': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'accuracy': (y_true == y_pred).mean(),
            'n_trades': len(y_true),
            'win_rate': y_true.mean()
        }

        # Financial metrics (if we have y_return)
        if 'y_return' in df.columns:
            returns = df['y_return'].values
            daily_returns = self._aggregate_to_daily_returns(df['Date'], returns)

            if len(daily_returns) > 0:
                sharpe = self._calculate_sharpe(daily_returns)
                metrics['sharpe'] = sharpe
                metrics['total_return'] = returns.sum()
                metrics['mean_return'] = returns.mean()
                metrics['std_return'] = returns.std()
            else:
                metrics['sharpe'] = 0.0
                metrics['total_return'] = 0.0
                metrics['mean_return'] = 0.0
                metrics['std_return'] = 0.0

        return metrics

    def _aggregate_to_daily_returns(
        self,
        dates: pd.Series,
        returns: np.ndarray
    ) -> np.ndarray:
        """Aggregate trade returns to daily returns."""
        df = pd.DataFrame({'Date': pd.to_datetime(dates), 'Return': returns})
        daily = df.groupby(df['Date'].dt.date)['Return'].sum()
        return daily.values

    def _calculate_sharpe(self, daily_returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(daily_returns) == 0 or daily_returns.std() == 0:
            return 0.0

        return (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    def _calculate_wfe(self, train_metrics: Dict, test_metrics: Dict) -> float:
        """
        Calculate Walk-Forward Efficiency (WFE).

        WFE = out_of_sample_performance / in_sample_performance

        Args:
            train_metrics: Training period metrics
            test_metrics: Test period metrics

        Returns:
            WFE ratio
        """
        # Use Sharpe ratio for WFE if available, otherwise use AUC
        if 'sharpe' in train_metrics and train_metrics['sharpe'] != 0:
            return test_metrics['sharpe'] / train_metrics['sharpe']
        elif train_metrics['auc'] != 0:
            return test_metrics['auc'] / train_metrics['auc']
        else:
            return 0.0

    def _calculate_aggregate_metrics(self):
        """Calculate aggregate metrics across all walk-forward windows."""
        logger.info("Calculating aggregate metrics across all windows...")

        # Extract metrics from each window
        test_sharpes = [w['test_metrics'].get('sharpe', 0) for w in self.window_results]
        test_aucs = [w['test_metrics']['auc'] for w in self.window_results]
        wfes = [w['wfe'] for w in self.window_results]

        logger.info("Aggregate Walk-Forward Results:")
        logger.info(f"  Mean Test Sharpe:  {np.mean(test_sharpes):.3f} ± {np.std(test_sharpes):.3f}")
        logger.info(f"  Mean Test AUC:     {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
        logger.info(f"  Mean WFE:          {np.mean(wfes):.2%} ± {np.std(wfes):.2%}")
        logger.info(f"  Min WFE:           {np.min(wfes):.2%}")
        logger.info(f"  Positive Windows:  {sum(s > 0 for s in test_sharpes)}/{len(test_sharpes)}")

    def get_results_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of walk-forward results."""
        summary_data = []

        for window in self.window_results:
            summary_data.append({
                'Window': window['window_name'],
                'Train_Start': window['train_start'],
                'Train_End': window['train_end'],
                'Test_Start': window['test_start'],
                'Test_End': window['test_end'],
                'Train_Samples': window['train_samples'],
                'Test_Samples': window['test_samples'],
                'CV_AUC': window['cv_score'],
                'Test_AUC': window['test_metrics']['auc'],
                'Test_Sharpe': window['test_metrics'].get('sharpe', 0),
                'WFE': window['wfe'],
                'Win_Rate': window['test_metrics']['win_rate']
            })

        return pd.DataFrame(summary_data)
