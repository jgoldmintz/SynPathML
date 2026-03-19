"""
Stage 1: Feature Selection & Importance
Runs L1/L2 Logistic Regression, Elastic Net, Random Forest, and SHAP analysis.
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from data_loader import DataLoader
from utils import (
    setup_logging,
    compute_class_weights,
    save_json,
    plot_feature_importance,
    plot_shap_summary,
)

warnings.filterwarnings("ignore", category=UserWarning)


class FeatureSelector:
    """
    Feature selection using multiple methods:
    - L1 Logistic Regression (Lasso): Sparse feature selection
    - L2 Logistic Regression (Ridge): Stable importance ranking
    - Elastic Net: Grouped feature selection
    - Random Forest: Non-linear importance with SHAP
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        cv_folds: int = 5,
        random_state: int = 42,
        logger=None
    ):
        """
        Initialize FeatureSelector.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            feature_names: List of feature names
            cv_folds: Number of cross-validation folds
            random_state: Random seed
            logger: Optional logger instance
        """
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.logger = logger

        # Standardize features for logistic regression
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)

        # Compute class weights
        self.class_weights = compute_class_weights(y)

        # Results storage
        self.results = {}

        if self.logger:
            self.logger.info(f"FeatureSelector initialized with {X.shape[0]} samples, {X.shape[1]} features")
            self.logger.info(f"Class distribution: {np.bincount(y)}")
            self.logger.info(f"Class weights: {self.class_weights}")

    def run_l1_logistic(
        self,
        Cs: int = 10,
        max_iter: int = 5000
    ) -> Dict:
        """
        Run L1 Logistic Regression (Lasso) with cross-validation.

        L1 regularization drives irrelevant feature weights to exactly zero,
        providing sparse feature selection.

        Args:
            Cs: Number of regularization strengths to try
            max_iter: Maximum iterations

        Returns:
            Dictionary with coefficients, selected features, and CV scores
        """
        if self.logger:
            self.logger.info("Running L1 Logistic Regression...")

        model = LogisticRegressionCV(
            penalty="l1",
            solver="saga",
            Cs=Cs,
            cv=self.cv_folds,
            class_weight="balanced",
            max_iter=max_iter,
            random_state=self.random_state,
            n_jobs=-1
        )

        model.fit(self.X_scaled, self.y)

        coefficients = model.coef_[0]
        selected_mask = coefficients != 0
        selected_features = [
            self.feature_names[i]
            for i in range(len(self.feature_names))
            if selected_mask[i]
        ]

        # Get CV scores at best C
        best_c_idx = np.where(model.Cs_ == model.C_[0])[0][0]
        cv_scores = model.scores_[1][:, best_c_idx]  # Scores for positive class

        result = {
            "coefficients": coefficients,
            "selected_features": selected_features,
            "n_selected": len(selected_features),
            "best_C": float(model.C_[0]),
            "cv_scores_mean": float(cv_scores.mean()),
            "cv_scores_std": float(cv_scores.std()),
            "model": model
        }

        self.results["l1"] = result

        if self.logger:
            self.logger.info(f"L1 selected {len(selected_features)} features")
            self.logger.info(f"Best C: {model.C_[0]:.4f}, CV score: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        return result

    def run_l2_logistic(
        self,
        Cs: int = 10,
        max_iter: int = 5000
    ) -> Dict:
        """
        Run L2 Logistic Regression (Ridge) with cross-validation.

        L2 regularization distributes weight across correlated features,
        providing more stable importance estimates.

        Args:
            Cs: Number of regularization strengths to try
            max_iter: Maximum iterations

        Returns:
            Dictionary with coefficients and importance ranking
        """
        if self.logger:
            self.logger.info("Running L2 Logistic Regression...")

        model = LogisticRegressionCV(
            penalty="l2",
            solver="lbfgs",
            Cs=Cs,
            cv=self.cv_folds,
            class_weight="balanced",
            max_iter=max_iter,
            random_state=self.random_state,
            n_jobs=-1
        )

        model.fit(self.X_scaled, self.y)

        coefficients = model.coef_[0]
        abs_coefficients = np.abs(coefficients)

        # Rank features by absolute coefficient
        ranking_indices = np.argsort(abs_coefficients)[::-1]
        importance_ranking = [self.feature_names[i] for i in ranking_indices]

        # Get CV scores
        best_c_idx = np.where(model.Cs_ == model.C_[0])[0][0]
        cv_scores = model.scores_[1][:, best_c_idx]

        result = {
            "coefficients": coefficients,
            "abs_coefficients": abs_coefficients,
            "importance_ranking": importance_ranking,
            "best_C": float(model.C_[0]),
            "cv_scores_mean": float(cv_scores.mean()),
            "cv_scores_std": float(cv_scores.std()),
            "model": model
        }

        self.results["l2"] = result

        if self.logger:
            self.logger.info(f"Best C: {model.C_[0]:.4f}, CV score: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
            self.logger.info(f"Top 5 features: {importance_ranking[:5]}")

        return result

    def run_elastic_net(
        self,
        l1_ratio: float = 0.5,
        alpha_range: Tuple[float, float] = (1e-4, 1e2),
        n_alphas: int = 20,
        max_iter: int = 5000
    ) -> Dict:
        """
        Run Elastic Net (combined L1 + L2) using SGDClassifier.

        Elastic Net combines L1 sparsity with L2 stability,
        effective for correlated feature groups.

        Args:
            l1_ratio: Mixing parameter (0 = L2, 1 = L1)
            alpha_range: Range of regularization strengths
            n_alphas: Number of alpha values to try
            max_iter: Maximum iterations

        Returns:
            Dictionary with coefficients and selected features
        """
        if self.logger:
            self.logger.info(f"Running Elastic Net (l1_ratio={l1_ratio})...")

        # Grid search over alpha
        alphas = np.logspace(
            np.log10(alpha_range[0]),
            np.log10(alpha_range[1]),
            n_alphas
        )

        best_score = -np.inf
        best_model = None
        best_alpha = None

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        for alpha in alphas:
            model = SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                l1_ratio=l1_ratio,
                alpha=alpha,
                class_weight="balanced",
                max_iter=max_iter,
                random_state=self.random_state,
                n_jobs=-1
            )

            scores = []
            for train_idx, val_idx in cv.split(self.X_scaled, self.y):
                model.fit(self.X_scaled[train_idx], self.y[train_idx])
                scores.append(model.score(self.X_scaled[val_idx], self.y[val_idx]))

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha
                best_model = model

        # Refit on full data with best alpha
        best_model = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            l1_ratio=l1_ratio,
            alpha=best_alpha,
            class_weight="balanced",
            max_iter=max_iter,
            random_state=self.random_state
        )
        best_model.fit(self.X_scaled, self.y)

        coefficients = best_model.coef_[0]
        selected_mask = coefficients != 0
        selected_features = [
            self.feature_names[i]
            for i in range(len(self.feature_names))
            if selected_mask[i]
        ]

        result = {
            "coefficients": coefficients,
            "selected_features": selected_features,
            "n_selected": len(selected_features),
            "best_alpha": float(best_alpha),
            "l1_ratio": l1_ratio,
            "cv_score": float(best_score),
            "model": best_model
        }

        self.results["elastic_net"] = result

        if self.logger:
            self.logger.info(f"Elastic Net selected {len(selected_features)} features")
            self.logger.info(f"Best alpha: {best_alpha:.4f}, CV score: {best_score:.4f}")

        return result

    def run_random_forest(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        min_samples_leaf: int = 5
    ) -> Dict:
        """
        Run Random Forest for non-linear feature importance.

        Random Forest captures feature interactions and non-linear relationships
        that linear models miss.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_leaf: Minimum samples per leaf

        Returns:
            Dictionary with feature importance and model
        """
        if self.logger:
            self.logger.info("Running Random Forest...")

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1
        )

        # Cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = []

        for train_idx, val_idx in cv.split(self.X, self.y):
            model.fit(self.X[train_idx], self.y[train_idx])
            cv_scores.append(model.score(self.X[val_idx], self.y[val_idx]))

        # Refit on full data
        model.fit(self.X, self.y)

        importance = model.feature_importances_
        ranking_indices = np.argsort(importance)[::-1]
        importance_ranking = [self.feature_names[i] for i in ranking_indices]

        result = {
            "importance": importance,
            "importance_ranking": importance_ranking,
            "cv_scores_mean": float(np.mean(cv_scores)),
            "cv_scores_std": float(np.std(cv_scores)),
            "model": model
        }

        self.results["random_forest"] = result

        if self.logger:
            self.logger.info(f"CV score: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
            self.logger.info(f"Top 5 features: {importance_ranking[:5]}")

        return result

    def compute_shap(
        self,
        model=None,
        max_samples: int = 1000
    ) -> Dict:
        """
        Compute SHAP values for feature importance.

        SHAP provides model-agnostic, theoretically grounded feature importance
        with per-sample attributions.

        Args:
            model: Model to explain (uses RF from results if None)
            max_samples: Maximum samples for SHAP computation

        Returns:
            Dictionary with SHAP values and mean absolute SHAP
        """
        try:
            import shap
        except ImportError:
            if self.logger:
                self.logger.warning("SHAP not installed, skipping SHAP analysis")
            return {"error": "SHAP not installed"}

        if self.logger:
            self.logger.info("Computing SHAP values...")

        if model is None:
            if "random_forest" not in self.results:
                raise ValueError("Must run random_forest first or provide a model")
            model = self.results["random_forest"]["model"]

        # Subsample for efficiency
        if len(self.X) > max_samples:
            np.random.seed(self.random_state)
            sample_idx = np.random.choice(len(self.X), max_samples, replace=False)
            X_sample = self.X[sample_idx]
        else:
            X_sample = self.X

        # TreeExplainer for Random Forest
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Handle binary classification (take positive class)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        ranking_indices = np.argsort(mean_abs_shap)[::-1]
        importance_ranking = [self.feature_names[i] for i in ranking_indices]

        result = {
            "shap_values": shap_values,
            "mean_abs_shap": mean_abs_shap,
            "importance_ranking": importance_ranking,
            "X_sample": X_sample
        }

        self.results["shap"] = result

        if self.logger:
            self.logger.info(f"Top 5 features by SHAP: {importance_ranking[:5]}")

        return result

    def run_all(self) -> Dict:
        """Run all feature selection methods."""
        self.run_l1_logistic()
        self.run_l2_logistic()
        self.run_elastic_net()
        self.run_random_forest()
        self.compute_shap()
        return self.results

    def get_importance_table(self) -> pd.DataFrame:
        """
        Create combined importance table from all methods.

        Returns:
            DataFrame with columns: Feature, L1_Coef, L2_Coef, ElasticNet_Coef,
            RF_Importance, SHAP_Mean
        """
        data = {"Feature": self.feature_names}

        if "l1" in self.results:
            data["L1_Coef"] = self.results["l1"]["coefficients"]
            data["L1_AbsCoef"] = np.abs(self.results["l1"]["coefficients"])

        if "l2" in self.results:
            data["L2_Coef"] = self.results["l2"]["coefficients"]
            data["L2_AbsCoef"] = self.results["l2"]["abs_coefficients"]

        if "elastic_net" in self.results:
            data["ElasticNet_Coef"] = self.results["elastic_net"]["coefficients"]
            data["ElasticNet_AbsCoef"] = np.abs(self.results["elastic_net"]["coefficients"])

        if "random_forest" in self.results:
            data["RF_Importance"] = self.results["random_forest"]["importance"]

        if "shap" in self.results and "mean_abs_shap" in self.results["shap"]:
            data["SHAP_Mean"] = self.results["shap"]["mean_abs_shap"]

        df = pd.DataFrame(data)

        # Add composite score (normalized average of available methods)
        importance_cols = [c for c in df.columns if c.endswith(("_AbsCoef", "_Importance", "_Mean"))]
        if importance_cols:
            # Normalize each column to [0, 1]
            normalized = df[importance_cols].apply(
                lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
            )
            df["Composite_Score"] = normalized.mean(axis=1)

        # Sort by composite score
        if "Composite_Score" in df.columns:
            df = df.sort_values("Composite_Score", ascending=False)

        return df

    def save_results(self, output_dir: str) -> None:
        """
        Save all results to files.

        Args:
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save importance table
        importance_df = self.get_importance_table()
        importance_df.to_csv(output_dir / "importance_table.tsv", sep="\t", index=False)

        # Save individual results (without models)
        results_summary = {}
        for method, result in self.results.items():
            results_summary[method] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in result.items()
                if k != "model" and not isinstance(v, np.ndarray) or k in ["coefficients", "importance", "mean_abs_shap"]
            }

        save_json(results_summary, output_dir / "feature_selection_results.json")

        # Save SHAP values
        if "shap" in self.results and "shap_values" in self.results["shap"]:
            np.save(
                output_dir / "shap_values.npy",
                self.results["shap"]["shap_values"]
            )

        # Save plots
        if "random_forest" in self.results:
            plot_feature_importance(
                importance_df,
                importance_col="RF_Importance",
                top_k=20,
                save_path=str(output_dir / "rf_importance.png"),
                title="Random Forest Feature Importance"
            )

        if "shap" in self.results and "shap_values" in self.results["shap"]:
            plot_shap_summary(
                self.results["shap"]["shap_values"],
                self.feature_names,
                self.results["shap"]["X_sample"],
                save_path=str(output_dir / "shap_summary.png")
            )

        # Save selected features from L1
        if "l1" in self.results:
            save_json(
                {"selected_features": self.results["l1"]["selected_features"]},
                output_dir / "l1_selected_features.json"
            )

        if self.logger:
            self.logger.info(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Feature Selection & Importance"
    )

    # Input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=str,
        help="Path to input TSV file or prepared data directory"
    )
    input_group.add_argument(
        "--sql",
        action="store_true",
        help="Load from SQL database (uses config file)"
    )

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration JSON file"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results"
    )

    # Options
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--rf-estimators",
        type=int,
        default=500,
        help="Number of Random Forest trees"
    )
    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=6,
        help="Maximum Random Forest tree depth"
    )
    parser.add_argument(
        "--elastic-net-l1-ratio",
        type=float,
        default=0.5,
        help="Elastic Net L1 ratio (0=L2, 1=L1)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.output_dir, "stage1_feature_selection")

    # Load data
    if args.input:
        input_path = Path(args.input)
        if input_path.is_dir():
            # Load prepared data
            X, y, is_synonymous, feature_names = DataLoader.load_prepared_data(input_path)
            logger.info(f"Loaded prepared data from {input_path}")
        else:
            # Load from TSV
            loader = DataLoader(config_path=args.config, logger=logger)
            loader.load_from_tsv(args.input)
            X, y, is_synonymous, feature_names = loader.prepare_features()
    else:
        # Load from SQL (uses config file)
        loader = DataLoader(config_path=args.config, logger=logger)
        loader.connect_sql()
        loader.load_from_sql()
        X, y, is_synonymous, feature_names = loader.prepare_features()

    # Run feature selection
    selector = FeatureSelector(
        X, y, feature_names,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        logger=logger
    )

    logger.info("=" * 50)
    logger.info("Running feature selection methods...")
    logger.info("=" * 50)

    selector.run_all()

    # Save results
    selector.save_results(args.output_dir)

    # Print summary
    importance_df = selector.get_importance_table()
    logger.info("\n" + "=" * 50)
    logger.info("Top 10 features by composite score:")
    logger.info("=" * 50)
    print(importance_df.head(10).to_string(index=False))

    logger.info("\nStage 1 complete")


if __name__ == "__main__":
    main()
