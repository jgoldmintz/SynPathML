#!/usr/bin/env python3
# SynPathML
# Copyright (C) 2023-2026  Jacob Goldmintz
# All rights reserved. See LICENSE for terms.

"""
Data loader for ML pipeline.
Supports loading from SQL database or TSV file.
Constructs unified feature matrix for training.

Configuration is loaded from external JSON file - no hardcoded values.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

from utils import setup_logging, save_json


# Default config file location
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to config file. If None, looks for config.json in same directory.

    Returns:
        Configuration dictionary

    Config file format:
    {
        "sql": {
            "host": "localhost",
            "port": 5432,
            "database": "mutations_db",
            "user": "username",
            "password": "password",
            "dialect": "postgresql",
            "table": "mutations",
            "query": null  // Optional: custom SQL query overrides table
        },
        "columns": {
            "pkey": "pkey",
            "label": "is_pathogenic",
            "is_synonymous": "is_synonymous",
            "gene": "gene"
        },
        "features": {
            "auto_detect": true,  // Auto-detect numeric columns as features
            "exclude": ["pkey", "gene", "chrom", "pos", "ref", "alt"],
            "include": null  // Optional: explicit list of feature columns (overrides auto-detect)
        }
    }
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    config_path = Path(config_path)

    if not config_path.exists():
        # Return minimal defaults if no config file
        return {
            "sql": {},
            "columns": {
                "pkey": "pkey",
                "label": "label",
                "is_synonymous": "is_synonymous",
                "gene": "gene"
            },
            "features": {
                "auto_detect": True,
                "exclude": ["pkey", "gene", "chrom", "pos", "ref", "alt"],
                "include": None
            }
        }

    with open(config_path, "r") as f:
        return json.load(f)


class DataLoader:
    """
    Load mutation data from SQL database or TSV file.
    Construct unified feature matrix for ML training.

    Configuration is loaded from external config.json file.
    Feature columns are auto-detected from database schema or DataFrame dtypes.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        config: Optional[Dict] = None,
        logger=None
    ):
        """
        Initialize DataLoader.

        Args:
            config_path: Path to configuration JSON file
            config: Configuration dict (overrides file config)
            logger: Optional logger instance
        """
        # Load config from file, then override with any passed config
        file_config = load_config(config_path)

        if config:
            # Deep merge
            for key in config:
                if key in file_config and isinstance(file_config[key], dict):
                    file_config[key].update(config[key])
                else:
                    file_config[key] = config[key]

        self.config = file_config
        self.logger = logger
        self.connection = None
        self.engine = None
        self.df = None
        self.feature_names = None
        self._schema_cache = None

    def connect_sql(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        dialect: Optional[str] = None
    ) -> None:
        """
        Establish SQL database connection.
        Parameters can be passed directly or loaded from config.

        Args:
            host: Database host (or from config)
            port: Database port (or from config)
            database: Database name (or from config)
            user: Username (or from config)
            password: Password (or from config)
            dialect: SQL dialect ('postgresql', 'mysql', 'sqlite') (or from config)
        """
        try:
            from sqlalchemy import create_engine, inspect
        except ImportError:
            raise ImportError("sqlalchemy required for SQL support: pip install sqlalchemy")

        # Get connection params from config if not provided
        sql_config = self.config.get("sql", {})
        host = host or sql_config.get("host")
        port = port or sql_config.get("port")
        database = database or sql_config.get("database")
        user = user or sql_config.get("user")
        password = password or sql_config.get("password")
        dialect = dialect or sql_config.get("dialect", "postgresql")

        if not all([host, database, user]):
            raise ValueError("Missing required SQL connection parameters. Provide via args or config.")

        if dialect == "postgresql":
            try:
                import psycopg2
            except ImportError:
                raise ImportError("psycopg2 required for PostgreSQL: pip install psycopg2-binary")
            url = f"postgresql://{user}:{password}@{host}:{port}/{database}"

        elif dialect == "mysql":
            try:
                import mysql.connector
            except ImportError:
                raise ImportError("mysql-connector required: pip install mysql-connector-python")
            url = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"

        elif dialect == "sqlite":
            url = f"sqlite:///{database}"

        else:
            raise ValueError(f"Unsupported dialect: {dialect}")

        self.engine = create_engine(url)
        self.connection = self.engine

        if self.logger:
            self.logger.info(f"Connected to {dialect} database at {host}:{port}/{database}")

    def get_table_schema(self, table_name: str) -> Dict[str, str]:
        """
        Get column names and types from database table.

        Args:
            table_name: Name of the table

        Returns:
            Dict mapping column name to SQL type string
        """
        if self.engine is None:
            raise RuntimeError("Must call connect_sql() first")

        try:
            from sqlalchemy import inspect
        except ImportError:
            raise ImportError("sqlalchemy required")

        inspector = inspect(self.engine)
        columns = inspector.get_columns(table_name)

        schema = {}
        for col in columns:
            col_type = str(col["type"]).upper()
            schema[col["name"]] = col_type

        self._schema_cache = schema
        return schema

    def detect_feature_columns_from_schema(
        self,
        table_name: Optional[str] = None,
        schema: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """
        Auto-detect feature columns from SQL table schema.
        Selects numeric columns, excludes metadata columns.

        Args:
            table_name: Table to inspect (or uses cached schema)
            schema: Pre-loaded schema dict

        Returns:
            List of feature column names
        """
        if schema is None:
            if self._schema_cache is not None:
                schema = self._schema_cache
            elif table_name:
                schema = self.get_table_schema(table_name)
            else:
                raise ValueError("Must provide table_name or schema")

        # Numeric SQL types
        numeric_types = {
            "INTEGER", "INT", "SMALLINT", "BIGINT",
            "REAL", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC",
            "FLOAT4", "FLOAT8", "INT4", "INT8", "INT2",
            "DOUBLE PRECISION"
        }

        # Get exclusion list from config
        feature_config = self.config.get("features", {})
        exclude = set(feature_config.get("exclude", []))

        # Add column config to exclusions
        col_config = self.config.get("columns", {})
        for key in ["pkey", "label", "is_synonymous", "gene"]:
            if key in col_config:
                exclude.add(col_config[key])

        # If explicit include list provided, use that
        include_list = feature_config.get("include")
        if include_list:
            return [c for c in include_list if c in schema]

        # Auto-detect numeric columns
        feature_cols = []
        for col_name, col_type in schema.items():
            if col_name.lower() in [e.lower() for e in exclude]:
                continue

            # Check if numeric type
            base_type = col_type.split("(")[0].strip()
            if base_type in numeric_types:
                feature_cols.append(col_name)

        return feature_cols

    def load_from_sql(
        self,
        table: Optional[str] = None,
        query: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load data from SQL database.
        Table name and query can be specified via args or config.

        Args:
            table: Table name (overrides config)
            query: Custom SQL query (overrides table-based loading)

        Returns:
            DataFrame with all data
        """
        if self.connection is None:
            raise RuntimeError("Must call connect_sql() first")

        sql_config = self.config.get("sql", {})

        # Query takes precedence, then arg, then config
        query = query or sql_config.get("query")

        if query:
            self.df = pd.read_sql(query, self.connection)
            if self.logger:
                self.logger.info(f"Loaded {len(self.df)} rows from custom query")
        else:
            table = table or sql_config.get("table")
            if not table:
                raise ValueError("Must provide table name via arg or config")

            # Get schema for later feature detection
            self.get_table_schema(table)

            self.df = pd.read_sql(f"SELECT * FROM {table}", self.connection)
            if self.logger:
                self.logger.info(f"Loaded {len(self.df)} rows from table '{table}'")

        return self.df

    @staticmethod
    def _read_single_tsv(filepath: Path) -> pd.DataFrame:
        """Read a single TSV/CSV with delimiter auto-detection."""
        with open(filepath, "r") as f:
            first_line = f.readline()
        if "\t" in first_line:
            sep = "\t"
        elif "," in first_line:
            sep = ","
        else:
            sep = "\t"
        return pd.read_csv(filepath, sep=sep)

    def _resolve_pipeline_prefix(self, stem: str) -> str:
        """
        Resolve filename stem to canonical pipeline prefix.

        Checks pipeline_map from config. Tries exact match, then strips
        leading gene name (e.g., 'SMN2.spliceai' -> tries 'spliceai').
        Falls back to stem with dots replaced by underscores.
        """
        tsv_dir_config = self.config.get("tsv_dir", {})
        pipeline_map = tsv_dir_config.get("pipeline_map", {})

        # Exact match
        if stem in pipeline_map:
            return pipeline_map[stem]

        # Strip leading gene name: GENE.pipeline.subtype -> try pipeline.subtype, then pipeline
        parts = stem.split(".")
        if len(parts) >= 2:
            for i in range(1, len(parts)):
                candidate = ".".join(parts[i:])
                if candidate in pipeline_map:
                    return pipeline_map[candidate]
            # No match in map — use non-gene portion
            return "_".join(parts[1:])

        return stem

    def _load_ground_truth(self, filepath: Path) -> pd.DataFrame:
        """
        Load ground truth CSV and normalize columns/values.

        Column mapping and value remapping are driven by
        config['tsv_dir']['ground_truth_columns'].
        """
        df = self._read_single_tsv(filepath)

        tsv_dir_config = self.config.get("tsv_dir", {})
        gt_col_map = tsv_dir_config.get("ground_truth_columns", {})

        # Rename columns per config mapping
        rename_map = {orig: target for orig, target in gt_col_map.items() if orig in df.columns}
        df = df.rename(columns=rename_map)

        col_config = self.config.get("columns", {})
        syn_col = col_config.get("is_synonymous", "is_synonymous")
        label_col = col_config.get("label", "is_pathogenic")

        # Remap is_synonymous: 2 -> 1 (synonymous), 1 -> 0 (nonsynonymous)
        if syn_col in df.columns:
            df[syn_col] = (df[syn_col] == 2).astype(int)

        # Remap label: 1 -> 1 (disease), 2 -> 0 (neutral)
        if label_col in df.columns:
            df[label_col] = (df[label_col] == 1).astype(int)

        # Keep only mapped columns that exist
        pkey_col = col_config.get("pkey", "pkey")
        gene_col = col_config.get("gene", "gene")
        keep = [c for c in [pkey_col, gene_col, syn_col, label_col] if c in df.columns]
        df = df[keep]

        if self.logger:
            self.logger.info(f"Ground truth: {len(df)} rows from {filepath.name}")

        return df

    def _is_ground_truth(self, filepath: Path) -> bool:
        """Check if a file is a ground truth file by inspecting its header."""
        tsv_dir_config = self.config.get("tsv_dir", {})
        gt_col_map = tsv_dir_config.get("ground_truth_columns", {})
        if not gt_col_map:
            return False

        with open(filepath, "r") as f:
            first_line = f.readline()

        return any(col_name in first_line for col_name in gt_col_map.keys())

    def load_from_tsv(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from a single TSV/CSV file.

        Args:
            filepath: Path to TSV file

        Returns:
            DataFrame with features and labels
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        self.df = self._read_single_tsv(filepath)

        if self.logger:
            self.logger.info(f"Loaded {len(self.df)} rows from {filepath}")

        return self.df

    def load_from_tsv_dir(self, directory: Union[str, Path]) -> pd.DataFrame:
        """
        Load multiple pipeline TSV files from a directory and join on pkey.

        Automatically detects the ground truth file by checking headers
        against config['tsv_dir']['ground_truth_columns']. Pipeline output
        columns are prefixed with the canonical pipeline name. Ground truth
        columns are mapped to standard names (pkey, is_pathogenic, etc.).

        Args:
            directory: Directory containing pipeline TSVs and ground truth CSV

        Returns:
            DataFrame with all pipeline features joined on pkey
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        col_config = self.config.get("columns", {})
        pkey_col = col_config.get("pkey", "pkey")

        # Discover all TSV/CSV files
        files = sorted(
            f for f in directory.iterdir()
            if f.suffix.lower() in (".tsv", ".csv", ".txt") and f.is_file()
        )
        if not files:
            raise FileNotFoundError(f"No TSV/CSV files found in {directory}")

        ground_truth_df = None
        pipeline_dfs = []

        for filepath in files:
            if self._is_ground_truth(filepath):
                ground_truth_df = self._load_ground_truth(filepath)
                continue

            df = self._read_single_tsv(filepath)
            if pkey_col not in df.columns:
                if self.logger:
                    self.logger.warning(f"Skipping {filepath.name}: no '{pkey_col}' column")
                continue

            prefix = self._resolve_pipeline_prefix(filepath.stem)
            rename_map = {col: f"{prefix}_{col}" for col in df.columns if col != pkey_col}
            df = df.rename(columns=rename_map)
            pipeline_dfs.append(df)

            if self.logger:
                self.logger.info(f"Loaded {filepath.name} -> prefix '{prefix}_' ({len(df)} rows, {len(df.columns)} cols)")

        if not pipeline_dfs:
            raise ValueError(f"No valid pipeline TSV files found in {directory}")

        # Outer join all pipeline outputs
        merged = pipeline_dfs[0]
        for df in pipeline_dfs[1:]:
            merged = merged.merge(df, on=pkey_col, how="outer")

        # Join ground truth (left join — only keep mutations with labels)
        if ground_truth_df is not None:
            merged = merged.merge(ground_truth_df, on=pkey_col, how="left")
            if self.logger:
                self.logger.info(f"Joined ground truth ({len(ground_truth_df)} labeled mutations)")
        else:
            if self.logger:
                self.logger.warning("No ground truth file detected in directory")

        self.df = merged

        if self.logger:
            self.logger.info(f"Merged {len(pipeline_dfs)} pipeline files -> {merged.shape[0]} rows, {merged.shape[1]} columns")

        return self.df

    def _detect_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Auto-detect feature columns from DataFrame.
        Uses numeric dtype detection and exclusion list from config.
        """
        feature_config = self.config.get("features", {})
        col_config = self.config.get("columns", {})

        # Build exclusion set
        exclude = set(feature_config.get("exclude", []))
        for key in ["pkey", "label", "is_synonymous", "gene"]:
            if key in col_config:
                exclude.add(col_config[key])

        # If explicit include list provided, use that
        include_list = feature_config.get("include")
        if include_list:
            return [c for c in include_list if c in df.columns]

        # Auto-detect: use schema cache if available, else DataFrame dtypes
        if self._schema_cache is not None:
            return self.detect_feature_columns_from_schema()

        # Auto-detect numeric columns from DataFrame
        numeric_dtypes = [np.float64, np.int64, np.float32, np.int32, np.float16, np.int16]

        feature_cols = []
        for col in df.columns:
            if col.lower() in [e.lower() for e in exclude]:
                continue

            if df[col].dtype in numeric_dtypes:
                feature_cols.append(col)

        return feature_cols

    def prepare_features(
        self,
        df: Optional[pd.DataFrame] = None,
        feature_columns: Optional[List[str]] = None,
        handle_missing: Literal["drop", "zero", "mean", "indicator"] = "indicator"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare feature matrix from DataFrame.
        Column names are read from config.

        Args:
            df: DataFrame to process (uses self.df if None)
            feature_columns: Explicit list of feature columns (auto-detect if None)
            handle_missing: Strategy for missing values:
                - 'drop': Drop rows with missing values
                - 'zero': Fill with zeros
                - 'mean': Fill with column mean
                - 'indicator': Fill with zero and add indicator columns

        Returns:
            Tuple of (X, y, is_synonymous, feature_names)
        """
        if df is None:
            df = self.df

        if df is None:
            raise RuntimeError("No data loaded. Call load_from_sql() or load_from_tsv() first")

        # Get column names from config
        col_config = self.config.get("columns", {})
        label_col = col_config.get("label", "label")
        syn_col = col_config.get("is_synonymous", "is_synonymous")

        # Get feature columns
        if feature_columns is None:
            feature_columns = self._detect_feature_columns(df)

        if self.logger:
            self.logger.info(f"Auto-detected {len(feature_columns)} feature columns")

        # Extract features
        X_df = df[feature_columns].copy()

        # Handle missing values
        if handle_missing == "drop":
            valid_mask = ~X_df.isnull().any(axis=1)
            X_df = X_df[valid_mask]
            df = df[valid_mask]

        elif handle_missing == "zero":
            X_df = X_df.fillna(0)

        elif handle_missing == "mean":
            X_df = X_df.fillna(X_df.mean())

        elif handle_missing == "indicator":
            # Add single column counting missing features per row
            X_df["feature_missing_count"] = X_df.isnull().sum(axis=1).astype(int)
            X_df = X_df.fillna(0)
            feature_columns = X_df.columns.tolist()

        # Convert to numpy
        X = X_df.values.astype(np.float32)

        # Extract labels
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found. Available: {list(df.columns)}")
        y = df[label_col].values.astype(np.int32)

        # Extract is_synonymous
        if syn_col in df.columns:
            is_synonymous = df[syn_col].values.astype(np.int32)
        else:
            if self.logger:
                self.logger.warning(f"Column '{syn_col}' not found, setting is_synonymous to zeros")
            is_synonymous = np.zeros(len(df), dtype=np.int32)

        self.feature_names = feature_columns

        if self.logger:
            self.logger.info(f"Feature matrix shape: {X.shape}")
            self.logger.info(f"Positive samples: {y.sum()} ({100*y.mean():.2f}%)")
            self.logger.info(f"Synonymous samples: {is_synonymous.sum()}")

        return X, y, is_synonymous, feature_columns

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        is_synonymous: Optional[np.ndarray] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        stratify: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Split data into train/validation/test sets with stratification.

        Args:
            X: Feature matrix
            y: Labels
            is_synonymous: Synonymous indicator (included in split)
            test_size: Fraction for test set
            val_size: Fraction for validation set (from remaining after test)
            random_state: Random seed
            stratify: Whether to stratify by label

        Returns:
            Dictionary with train/val/test splits for X, y, and is_synonymous
        """
        stratify_arr = y if stratify else None

        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arr
        )

        if is_synonymous is not None:
            _, _, syn_trainval, syn_test = train_test_split(
                X, is_synonymous,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_arr
            )
        else:
            syn_trainval = syn_test = None

        # Second split: train vs val
        val_fraction = val_size / (1 - test_size)
        stratify_trainval = y_trainval if stratify else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_fraction,
            random_state=random_state,
            stratify=stratify_trainval
        )

        if syn_trainval is not None:
            _, _, syn_train, syn_val = train_test_split(
                X_trainval, syn_trainval,
                test_size=val_fraction,
                random_state=random_state,
                stratify=stratify_trainval
            )
        else:
            syn_train = syn_val = None

        splits = {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }

        if syn_train is not None:
            splits["is_synonymous_train"] = syn_train
            splits["is_synonymous_val"] = syn_val
            splits["is_synonymous_test"] = syn_test

        if self.logger:
            self.logger.info(f"Train: {len(X_train)} samples ({y_train.sum()} positive)")
            self.logger.info(f"Val: {len(X_val)} samples ({y_val.sum()} positive)")
            self.logger.info(f"Test: {len(X_test)} samples ({y_test.sum()} positive)")

        return splits

    def get_cv_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 10,
        random_state: int = 42
    ) -> StratifiedKFold:
        """
        Get cross-validation splitter.

        Args:
            X: Feature matrix
            y: Labels
            n_splits: Number of CV folds
            random_state: Random seed

        Returns:
            StratifiedKFold splitter
        """
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )

    def get_pkeys(self, df: Optional[pd.DataFrame] = None) -> Optional[np.ndarray]:
        """Get pkey column from DataFrame."""
        if df is None:
            df = self.df

        col_config = self.config.get("columns", {})
        pkey_col = col_config.get("pkey", "pkey")

        if pkey_col in df.columns:
            return df[pkey_col].values
        return None

    def get_genes(self, df: Optional[pd.DataFrame] = None) -> Optional[np.ndarray]:
        """Get gene column from DataFrame."""
        if df is None:
            df = self.df

        col_config = self.config.get("columns", {})
        gene_col = col_config.get("gene", "gene")

        if gene_col in df.columns:
            return df[gene_col].values
        return None

    def save_prepared_data(
        self,
        output_dir: Union[str, Path],
        X: np.ndarray,
        y: np.ndarray,
        is_synonymous: np.ndarray,
        feature_names: List[str],
        splits: Optional[Dict] = None
    ) -> None:
        """
        Save prepared data to files.

        Args:
            output_dir: Output directory
            X: Feature matrix
            y: Labels
            is_synonymous: Synonymous indicator
            feature_names: Feature names
            splits: Optional train/val/test splits
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full data
        np.save(output_dir / "X.npy", X)
        np.save(output_dir / "y.npy", y)
        np.save(output_dir / "is_synonymous.npy", is_synonymous)

        save_json({"feature_names": feature_names}, output_dir / "feature_names.json")

        # Save splits if provided
        if splits:
            for key, arr in splits.items():
                np.save(output_dir / f"{key}.npy", arr)

        if self.logger:
            self.logger.info(f"Saved prepared data to {output_dir}")

    @staticmethod
    def load_prepared_data(
        data_dir: Union[str, Path]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Load previously prepared data.

        Args:
            data_dir: Directory containing saved data

        Returns:
            Tuple of (X, y, is_synonymous, feature_names)
        """
        data_dir = Path(data_dir)

        X = np.load(data_dir / "X.npy")
        y = np.load(data_dir / "y.npy")
        is_synonymous = np.load(data_dir / "is_synonymous.npy")

        with open(data_dir / "feature_names.json", "r") as f:
            feature_names = json.load(f)["feature_names"]

        return X, y, is_synonymous, feature_names


def main():
    parser = argparse.ArgumentParser(
        description="Load and prepare mutation data for ML pipeline"
    )

    # Input source
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--tsv",
        type=str,
        help="Path to input TSV file"
    )
    source_group.add_argument(
        "--sql",
        action="store_true",
        help="Load from SQL database (uses config file)"
    )
    source_group.add_argument(
        "--tsv-dir",
        type=str,
        help="Directory of pipeline TSV files + ground truth CSV (joined on pkey)"
    )

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration JSON file (default: config.json in same directory)"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for prepared data"
    )

    # Options
    parser.add_argument(
        "--missing-strategy",
        type=str,
        choices=["drop", "zero", "mean", "indicator"],
        default="indicator",
        help="Strategy for handling missing values"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction for test set"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Fraction for validation set"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.output_dir, "data_loader")

    # Load data
    loader = DataLoader(config_path=args.config, logger=logger)

    if args.tsv:
        loader.load_from_tsv(args.tsv)
    elif args.tsv_dir:
        loader.load_from_tsv_dir(args.tsv_dir)
    else:
        loader.connect_sql()  # Uses config
        loader.load_from_sql()  # Uses config

    # Prepare features (auto-detects columns from config and schema)
    X, y, is_synonymous, feature_names = loader.prepare_features(
        handle_missing=args.missing_strategy
    )

    # Split data
    splits = loader.split_data(
        X, y, is_synonymous,
        test_size=args.test_size,
        val_size=args.val_size
    )

    # Save
    loader.save_prepared_data(
        args.output_dir,
        X, y, is_synonymous, feature_names,
        splits=splits
    )

    logger.info("Data preparation complete")


if __name__ == "__main__":
    main()
