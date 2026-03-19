"""
Stage 2: Feature Analysis
Analyzes Stage 1 results to identify high-confidence features,
compare rankings, cluster correlated features, and stratify by mutation type.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from data_loader import DataLoader
from utils import setup_logging, save_json, load_json


class FeatureAnalyzer:
    """
    Analyze Stage 1 feature selection results:
    - Intersect high-confidence features from multiple methods
    - Compare linear vs non-linear importance rankings
    - Cluster correlated features into mechanism groups
    - Stratify importance by mutation type (synonymous vs nonsynonymous)
    """

    def __init__(
        self,
        stage1_results: Dict,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        is_synonymous: np.ndarray,
        logger=None
    ):
        """
        Initialize FeatureAnalyzer.

        Args:
            stage1_results: Results from Stage 1 feature selection
            X: Feature matrix
            y: Labels
            feature_names: List of feature names
            is_synonymous: Boolean array indicating synonymous mutations
            logger: Optional logger instance
        """
        self.stage1_results = stage1_results
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.is_synonymous = is_synonymous.astype(bool)
        self.logger = logger

        # Load importance table if available
        self.importance_df = self._build_importance_df()

        self.analysis_results = {}

        if self.logger:
            self.logger.info(f"FeatureAnalyzer initialized with {len(feature_names)} features")
            self.logger.info(f"Synonymous: {self.is_synonymous.sum()}, Nonsynonymous: {(~self.is_synonymous).sum()}")

    def _build_importance_df(self) -> pd.DataFrame:
        """Build importance DataFrame from Stage 1 results."""
        data = {"Feature": self.feature_names}

        if "l1" in self.stage1_results:
            coefs = self.stage1_results["l1"].get("coefficients", [])
            if len(coefs) == len(self.feature_names):
                data["L1_Coef"] = np.abs(coefs)

        if "l2" in self.stage1_results:
            coefs = self.stage1_results["l2"].get("coefficients", [])
            if len(coefs) == len(self.feature_names):
                data["L2_Coef"] = np.abs(coefs)

        if "elastic_net" in self.stage1_results:
            coefs = self.stage1_results["elastic_net"].get("coefficients", [])
            if len(coefs) == len(self.feature_names):
                data["ElasticNet_Coef"] = np.abs(coefs)

        if "random_forest" in self.stage1_results:
            importance = self.stage1_results["random_forest"].get("importance", [])
            if len(importance) == len(self.feature_names):
                data["RF_Importance"] = importance

        if "shap" in self.stage1_results:
            shap_mean = self.stage1_results["shap"].get("mean_abs_shap", [])
            if len(shap_mean) == len(self.feature_names):
                data["SHAP_Mean"] = shap_mean

        return pd.DataFrame(data)

    def intersect_features(
        self,
        l1_threshold: float = 0.0,
        rf_top_k: int = 20,
        require_both: bool = True
    ) -> List[str]:
        """
        Find high-confidence features that appear in both L1 selection and top RF.

        Args:
            l1_threshold: Minimum absolute L1 coefficient to be considered selected
            rf_top_k: Number of top RF features to consider
            require_both: If True, require feature in both sets; else union

        Returns:
            List of high-confidence feature names
        """
        if self.logger:
            self.logger.info(f"Finding high-confidence features (L1 threshold={l1_threshold}, RF top-{rf_top_k})")

        l1_selected = set()
        rf_top = set()

        # L1 selected features
        if "L1_Coef" in self.importance_df.columns:
            l1_mask = self.importance_df["L1_Coef"] > l1_threshold
            l1_selected = set(self.importance_df[l1_mask]["Feature"].tolist())

        # RF top-k features
        if "RF_Importance" in self.importance_df.columns:
            rf_sorted = self.importance_df.nlargest(rf_top_k, "RF_Importance")
            rf_top = set(rf_sorted["Feature"].tolist())

        if require_both:
            high_confidence = sorted(l1_selected & rf_top)
        else:
            high_confidence = sorted(l1_selected | rf_top)

        self.analysis_results["high_confidence_features"] = high_confidence
        self.analysis_results["l1_selected"] = sorted(l1_selected)
        self.analysis_results["rf_top"] = sorted(rf_top)

        if self.logger:
            self.logger.info(f"L1 selected: {len(l1_selected)} features")
            self.logger.info(f"RF top-{rf_top_k}: {len(rf_top)} features")
            self.logger.info(f"Intersection: {len(high_confidence)} features")
            if high_confidence:
                self.logger.info(f"High-confidence features: {high_confidence}")

        return high_confidence

    def compare_rankings(self) -> pd.DataFrame:
        """
        Compare linear (L1/L2) vs non-linear (RF/SHAP) importance rankings.

        Features with high RF/SHAP but low linear importance may involve
        non-linear effects or interactions.

        Returns:
            DataFrame with ranking comparisons and discrepancy scores
        """
        if self.logger:
            self.logger.info("Comparing linear vs non-linear rankings...")

        df = self.importance_df.copy()

        # Compute ranks for each method
        rank_cols = []
        for col in ["L1_Coef", "L2_Coef", "RF_Importance", "SHAP_Mean"]:
            if col in df.columns:
                rank_col = f"{col}_Rank"
                df[rank_col] = df[col].rank(ascending=False)
                rank_cols.append(rank_col)

        # Linear average rank
        linear_ranks = [c for c in rank_cols if "L1" in c or "L2" in c]
        if linear_ranks:
            df["Linear_Avg_Rank"] = df[linear_ranks].mean(axis=1)

        # Non-linear average rank
        nonlinear_ranks = [c for c in rank_cols if "RF" in c or "SHAP" in c]
        if nonlinear_ranks:
            df["Nonlinear_Avg_Rank"] = df[nonlinear_ranks].mean(axis=1)

        # Discrepancy: positive = higher in non-linear than linear (interaction candidate)
        if "Linear_Avg_Rank" in df.columns and "Nonlinear_Avg_Rank" in df.columns:
            df["Rank_Discrepancy"] = df["Linear_Avg_Rank"] - df["Nonlinear_Avg_Rank"]

        # Compute Spearman correlations between methods
        correlations = {}
        importance_cols = [c for c in ["L1_Coef", "L2_Coef", "RF_Importance", "SHAP_Mean"] if c in df.columns]

        for i, col1 in enumerate(importance_cols):
            for col2 in importance_cols[i+1:]:
                rho, pval = stats.spearmanr(df[col1], df[col2])
                correlations[f"{col1}_vs_{col2}"] = {"spearman_rho": rho, "pvalue": pval}

        self.analysis_results["ranking_comparison"] = df
        self.analysis_results["method_correlations"] = correlations

        # Identify interaction candidates (high discrepancy)
        if "Rank_Discrepancy" in df.columns:
            interaction_candidates = df.nlargest(10, "Rank_Discrepancy")["Feature"].tolist()
            self.analysis_results["interaction_candidates"] = interaction_candidates

            if self.logger:
                self.logger.info(f"Interaction candidates (high nonlinear, low linear): {interaction_candidates[:5]}")

        return df

    def cluster_features(
        self,
        correlation_threshold: float = 0.7,
        method: str = "average"
    ) -> Dict[str, List[str]]:
        """
        Cluster correlated features into mechanism groups.

        Uses hierarchical clustering on feature correlation matrix.

        Args:
            correlation_threshold: Minimum correlation to be in same cluster
            method: Linkage method ('average', 'complete', 'single')

        Returns:
            Dictionary mapping cluster ID to list of feature names
        """
        if self.logger:
            self.logger.info(f"Clustering features (correlation threshold={correlation_threshold})...")

        # Compute correlation matrix
        corr_matrix = np.corrcoef(self.X.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0)

        # Convert to distance matrix
        distance_matrix = 1 - np.abs(corr_matrix)
        np.fill_diagonal(distance_matrix, 0)

        # Hierarchical clustering
        condensed_dist = squareform(distance_matrix, checks=False)
        linkage_matrix = hierarchy.linkage(condensed_dist, method=method)

        # Cut tree at threshold
        distance_threshold = 1 - correlation_threshold
        cluster_labels = hierarchy.fcluster(linkage_matrix, distance_threshold, criterion="distance")

        # Group features by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            cluster_id = f"cluster_{label}"
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(self.feature_names[i])

        # Sort clusters by size
        clusters = dict(sorted(clusters.items(), key=lambda x: -len(x[1])))

        self.analysis_results["feature_clusters"] = clusters
        self.analysis_results["n_clusters"] = len(clusters)

        if self.logger:
            self.logger.info(f"Found {len(clusters)} clusters")
            for cid, features in list(clusters.items())[:5]:
                self.logger.info(f"  {cid}: {features[:3]}{'...' if len(features) > 3 else ''}")

        return clusters

    def stratify_by_mutation_type(self) -> Dict:
        """
        Analyze feature importance separately for synonymous vs nonsynonymous.

        This informs gating/attention design for Stage 3 neural network.

        Returns:
            Dictionary with separate importance analyses
        """
        if self.logger:
            self.logger.info("Stratifying importance by mutation type...")

        from sklearn.ensemble import RandomForestClassifier

        results = {}

        # Synonymous subset
        syn_mask = self.is_synonymous
        if syn_mask.sum() > 10:
            X_syn = self.X[syn_mask]
            y_syn = self.y[syn_mask]

            rf_syn = RandomForestClassifier(
                n_estimators=200,
                max_depth=4,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
            rf_syn.fit(X_syn, y_syn)

            syn_importance = pd.DataFrame({
                "Feature": self.feature_names,
                "Importance_Syn": rf_syn.feature_importances_
            }).sort_values("Importance_Syn", ascending=False)

            results["synonymous"] = {
                "n_samples": int(syn_mask.sum()),
                "n_positive": int(y_syn.sum()),
                "importance": syn_importance.to_dict(orient="records"),
                "top_features": syn_importance.head(10)["Feature"].tolist()
            }

            if self.logger:
                self.logger.info(f"Synonymous top 5: {results['synonymous']['top_features'][:5]}")

        # Nonsynonymous subset
        nonsyn_mask = ~self.is_synonymous
        if nonsyn_mask.sum() > 10:
            X_nonsyn = self.X[nonsyn_mask]
            y_nonsyn = self.y[nonsyn_mask]

            rf_nonsyn = RandomForestClassifier(
                n_estimators=200,
                max_depth=4,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
            rf_nonsyn.fit(X_nonsyn, y_nonsyn)

            nonsyn_importance = pd.DataFrame({
                "Feature": self.feature_names,
                "Importance_Nonsyn": rf_nonsyn.feature_importances_
            }).sort_values("Importance_Nonsyn", ascending=False)

            results["nonsynonymous"] = {
                "n_samples": int(nonsyn_mask.sum()),
                "n_positive": int(y_nonsyn.sum()),
                "importance": nonsyn_importance.to_dict(orient="records"),
                "top_features": nonsyn_importance.head(10)["Feature"].tolist()
            }

            if self.logger:
                self.logger.info(f"Nonsynonymous top 5: {results['nonsynonymous']['top_features'][:5]}")

        # Compare: which features differ most between syn and nonsyn?
        if "synonymous" in results and "nonsynonymous" in results:
            syn_df = pd.DataFrame(results["synonymous"]["importance"])
            nonsyn_df = pd.DataFrame(results["nonsynonymous"]["importance"])

            merged = syn_df.merge(nonsyn_df, on="Feature")
            merged["Importance_Diff"] = merged["Importance_Syn"] - merged["Importance_Nonsyn"]

            # Features more important for synonymous
            syn_specific = merged.nlargest(5, "Importance_Diff")["Feature"].tolist()

            # Features more important for nonsynonymous
            nonsyn_specific = merged.nsmallest(5, "Importance_Diff")["Feature"].tolist()

            results["comparison"] = {
                "synonymous_specific": syn_specific,
                "nonsynonymous_specific": nonsyn_specific
            }

            if self.logger:
                self.logger.info(f"Synonymous-specific features: {syn_specific}")
                self.logger.info(f"Nonsynonymous-specific features: {nonsyn_specific}")

        self.analysis_results["stratified_importance"] = results
        return results

    def identify_mechanism_groups(self) -> Dict[str, List[str]]:
        """
        Identify mechanism groups based on feature naming conventions.

        Groups features by their biological mechanism (splice, RNA, miRNA, etc.)
        for use with attention-based neural network.

        Returns:
            Dictionary mapping mechanism name to feature indices
        """
        mechanism_prefixes = {
            "splicing": ["spliceai_", "genesplicer_"],
            "rna_structure": ["rnafold_"],
            "mirna": ["miranda_"],
            "glycosylation": ["netnglyc_"],
            "phosphorylation": ["netphos_"],
            "protein_structure": ["netsurfp_"],
            "evolution": ["evmutation_"],
            "immune": ["netmhc_"],
        }

        mechanism_groups = {}
        mechanism_indices = {}

        for mechanism, prefixes in mechanism_prefixes.items():
            features = []
            indices = []
            for i, feat in enumerate(self.feature_names):
                feat_lower = feat.lower()
                if any(feat_lower.startswith(p) for p in prefixes):
                    features.append(feat)
                    indices.append(i)

            if features:
                mechanism_groups[mechanism] = features
                mechanism_indices[mechanism] = indices

        # Catch remaining features as "other"
        assigned = set()
        for features in mechanism_groups.values():
            assigned.update(features)

        other_features = [f for f in self.feature_names if f not in assigned]
        other_indices = [i for i, f in enumerate(self.feature_names) if f not in assigned]

        if other_features:
            mechanism_groups["other"] = other_features
            mechanism_indices["other"] = other_indices

        self.analysis_results["mechanism_groups"] = mechanism_groups
        self.analysis_results["mechanism_indices"] = mechanism_indices

        if self.logger:
            self.logger.info("Mechanism groups:")
            for mech, features in mechanism_groups.items():
                self.logger.info(f"  {mech}: {len(features)} features")

        return mechanism_indices

    def recommend_features(
        self,
        max_features: Optional[int] = None,
        min_importance_percentile: float = 50
    ) -> List[str]:
        """
        Recommend features for Stage 3 neural network.

        Prioritizes high-confidence features, then adds top-ranked features
        up to max_features.

        Args:
            max_features: Maximum number of features to recommend (None = all features)
            min_importance_percentile: Minimum percentile threshold

        Returns:
            List of recommended feature names
        """
        # Default to all features if not specified
        if max_features is None:
            max_features = len(self.feature_names)

        recommended = set()

        # Start with high-confidence features
        if "high_confidence_features" in self.analysis_results:
            recommended.update(self.analysis_results["high_confidence_features"])

        # Add top features by composite score
        if len(recommended) < max_features and "RF_Importance" in self.importance_df.columns:
            threshold = np.percentile(
                self.importance_df["RF_Importance"],
                min_importance_percentile
            )
            above_threshold = self.importance_df[
                self.importance_df["RF_Importance"] >= threshold
            ]["Feature"].tolist()

            for feat in above_threshold:
                if len(recommended) >= max_features:
                    break
                recommended.add(feat)

        recommended = sorted(recommended)
        self.analysis_results["recommended_features"] = recommended

        if self.logger:
            self.logger.info(f"Recommended {len(recommended)} features for Stage 3")

        return recommended

    def generate_report(self) -> str:
        """Generate markdown analysis report."""
        lines = [
            "# Stage 2: Feature Analysis Report",
            "",
            "## Summary",
            f"- Total features analyzed: {len(self.feature_names)}",
            f"- Total samples: {len(self.y)}",
            f"- Positive samples: {self.y.sum()} ({100*self.y.mean():.2f}%)",
            f"- Synonymous samples: {self.is_synonymous.sum()}",
            f"- Nonsynonymous samples: {(~self.is_synonymous).sum()}",
            "",
        ]

        # High-confidence features
        if "high_confidence_features" in self.analysis_results:
            features = self.analysis_results["high_confidence_features"]
            lines.extend([
                "## High-Confidence Features",
                f"Features appearing in both L1 selection and RF top-20: {len(features)}",
                "",
            ])
            for f in features:
                lines.append(f"- {f}")
            lines.append("")

        # Interaction candidates
        if "interaction_candidates" in self.analysis_results:
            candidates = self.analysis_results["interaction_candidates"]
            lines.extend([
                "## Interaction Candidates",
                "Features with high non-linear but low linear importance (may involve interactions):",
                "",
            ])
            for f in candidates[:5]:
                lines.append(f"- {f}")
            lines.append("")

        # Mechanism groups
        if "mechanism_groups" in self.analysis_results:
            groups = self.analysis_results["mechanism_groups"]
            lines.extend([
                "## Mechanism Groups",
                f"Identified {len(groups)} mechanism groups for attention-based NN:",
                "",
            ])
            for mech, features in groups.items():
                lines.append(f"- **{mech}**: {len(features)} features")
            lines.append("")

        # Stratified analysis
        if "stratified_importance" in self.analysis_results:
            strat = self.analysis_results["stratified_importance"]
            lines.extend([
                "## Mutation Type Stratification",
                "",
            ])

            if "synonymous" in strat:
                lines.append(f"### Synonymous (n={strat['synonymous']['n_samples']}, {strat['synonymous']['n_positive']} positive)")
                lines.append("Top features: " + ", ".join(strat["synonymous"]["top_features"][:5]))
                lines.append("")

            if "nonsynonymous" in strat:
                lines.append(f"### Nonsynonymous (n={strat['nonsynonymous']['n_samples']}, {strat['nonsynonymous']['n_positive']} positive)")
                lines.append("Top features: " + ", ".join(strat["nonsynonymous"]["top_features"][:5]))
                lines.append("")

            if "comparison" in strat:
                lines.append("### Differential Features")
                lines.append(f"Synonymous-specific: {strat['comparison']['synonymous_specific']}")
                lines.append(f"Nonsynonymous-specific: {strat['comparison']['nonsynonymous_specific']}")
                lines.append("")

        # Recommended features
        if "recommended_features" in self.analysis_results:
            features = self.analysis_results["recommended_features"]
            lines.extend([
                "## Recommended Features for Stage 3",
                f"Total: {len(features)} features",
                "",
            ])
            for f in features:
                lines.append(f"- {f}")
            lines.append("")

        # Recommendations for NN architecture
        lines.extend([
            "## Recommendations for Stage 3 Neural Network",
            "",
        ])

        if "mechanism_groups" in self.analysis_results:
            lines.append("- **Attention-based architecture**: Use mechanism groups for interpretable attention")

        if "stratified_importance" in self.analysis_results and "comparison" in self.analysis_results["stratified_importance"]:
            lines.append("- **Gated architecture**: Mutation type stratification shows differential feature importance")
            lines.append("  - Use is_synonymous flag to gate between regulatory and protein features")

        if "interaction_candidates" in self.analysis_results:
            lines.append("- **Non-linear layers**: Interaction candidates suggest non-linear effects")

        return "\n".join(lines)

    def run_all(self) -> Dict:
        """Run all analysis methods."""
        self.intersect_features()
        self.compare_rankings()
        self.cluster_features()
        self.stratify_by_mutation_type()
        self.identify_mechanism_groups()
        self.recommend_features()
        return self.analysis_results

    def save_results(self, output_dir: str) -> None:
        """Save all results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save high-confidence features
        if "high_confidence_features" in self.analysis_results:
            save_json(
                {"high_confidence_features": self.analysis_results["high_confidence_features"]},
                output_dir / "high_confidence_features.json"
            )

        # Save ranking comparison
        if "ranking_comparison" in self.analysis_results:
            self.analysis_results["ranking_comparison"].to_csv(
                output_dir / "ranking_comparison.tsv",
                sep="\t",
                index=False
            )

        # Save feature clusters
        if "feature_clusters" in self.analysis_results:
            save_json(
                self.analysis_results["feature_clusters"],
                output_dir / "feature_clusters.json"
            )

        # Save mechanism groups
        if "mechanism_indices" in self.analysis_results:
            save_json(
                self.analysis_results["mechanism_indices"],
                output_dir / "mechanism_indices.json"
            )

        # Save stratified importance
        if "stratified_importance" in self.analysis_results:
            save_json(
                self.analysis_results["stratified_importance"],
                output_dir / "stratified_importance.json"
            )

        # Save recommended features
        if "recommended_features" in self.analysis_results:
            save_json(
                {"recommended_features": self.analysis_results["recommended_features"]},
                output_dir / "recommended_features.json"
            )

        # Save full results
        # Convert non-serializable objects
        results_serializable = {}
        for k, v in self.analysis_results.items():
            if isinstance(v, pd.DataFrame):
                results_serializable[k] = v.to_dict(orient="records")
            elif isinstance(v, np.ndarray):
                results_serializable[k] = v.tolist()
            else:
                results_serializable[k] = v

        save_json(results_serializable, output_dir / "stage2_results.json")

        # Save report
        report = self.generate_report()
        with open(output_dir / "analysis_report.md", "w") as f:
            f.write(report)

        if self.logger:
            self.logger.info(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Feature Analysis"
    )

    # Input
    parser.add_argument(
        "--stage1-results",
        type=str,
        required=True,
        help="Path to Stage 1 results directory"
    )

    # Data source
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--input",
        type=str,
        help="Path to input TSV file or prepared data directory"
    )
    data_group.add_argument(
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
        "--correlation-threshold",
        type=float,
        default=0.7,
        help="Correlation threshold for feature clustering"
    )
    parser.add_argument(
        "--rf-top-k",
        type=int,
        default=20,
        help="Number of top RF features for intersection"
    )
    parser.add_argument(
        "--max-recommended",
        type=int,
        default=None,
        help="Maximum features to recommend for Stage 3 (default: all features)"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.output_dir, "stage2_analysis")

    # Load Stage 1 results
    stage1_results_path = Path(args.stage1_results) / "feature_selection_results.json"
    if not stage1_results_path.exists():
        raise FileNotFoundError(f"Stage 1 results not found: {stage1_results_path}")

    stage1_results = load_json(stage1_results_path)
    logger.info(f"Loaded Stage 1 results from {stage1_results_path}")

    # Load data
    if args.input:
        input_path = Path(args.input)
        if input_path.is_dir():
            X, y, is_synonymous, feature_names = DataLoader.load_prepared_data(input_path)
        else:
            loader = DataLoader(config_path=args.config, logger=logger)
            loader.load_from_tsv(args.input)
            X, y, is_synonymous, feature_names = loader.prepare_features()
    else:
        # Load from SQL (uses config file)
        loader = DataLoader(config_path=args.config, logger=logger)
        loader.connect_sql()
        loader.load_from_sql()
        X, y, is_synonymous, feature_names = loader.prepare_features()

    # Run analysis
    analyzer = FeatureAnalyzer(
        stage1_results,
        X, y, feature_names, is_synonymous,
        logger=logger
    )

    logger.info("=" * 50)
    logger.info("Running feature analysis...")
    logger.info("=" * 50)

    analyzer.intersect_features(rf_top_k=args.rf_top_k)
    analyzer.compare_rankings()
    analyzer.cluster_features(correlation_threshold=args.correlation_threshold)
    analyzer.stratify_by_mutation_type()
    analyzer.identify_mechanism_groups()
    analyzer.recommend_features(max_features=args.max_recommended)

    # Save results
    analyzer.save_results(args.output_dir)

    # Print report
    report = analyzer.generate_report()
    print("\n" + report)

    logger.info("\nStage 2 complete")


if __name__ == "__main__":
    main()
