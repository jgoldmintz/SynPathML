# SynPathML: Disease-Associated Mutation Prediction Pipeline

A three-stage machine learning pipeline for predicting disease-associated synonymous mutations using BioFeatureFactory features.

## Overview

This pipeline implements a staged approach to mutation pathogenicity prediction:

| Stage | Purpose | Methods |
|-------|---------|---------|
| 1 | Feature Selection | L1/L2 Logistic Regression, Elastic Net, Random Forest, SHAP |
| 2 | Analysis | Feature intersection, clustering, mechanism grouping, stratification |
| 3 | Neural Network | BaseNN, AttentionNN, GatedNN with PyTorch |

The pipeline uses **joint training** on all mutations (synonymous + nonsynonymous) to leverage shared regulatory signals, then evaluates performance specifically on synonymous mutations.

## Requirements

```
numpy
pandas
scikit-learn
scipy
torch
shap
matplotlib
sqlalchemy (for SQL database input)
psycopg2 or pymysql (database driver)
```

## Directory Structure

```
SynPathML/
├── data_loader.py              # Data loading from SQL/TSV
├── utils.py                    # Shared utilities (metrics, plotting, logging)
├── stage1_feature_selection.py # Feature importance analysis
├── stage2_analysis.py          # Feature relationship analysis
├── stage3_neural_network.py    # Neural network training
├── config.template.json        # Configuration template
├── stage_explanations.txt      # Detailed documentation
└── README.md
```

## Configuration

Copy `config.template.json` to `config.json` and edit:

```json
{
    "sql": {
        "host": "localhost",
        "port": 5432,
        "database": "your_database",
        "user": "your_username",
        "password": "your_password",
        "dialect": "postgresql",
        "table": "mutations"
    },
    "columns": {
        "pkey": "pkey",
        "label": "is_pathogenic",
        "is_synonymous": "is_synonymous",
        "gene": "gene"
    },
    "features": {
        "auto_detect": true,
        "exclude": ["pkey", "gene", "chrom", "pos", "ref", "alt"]
    }
}
```

**SQL section**: Required for database connection.

**Columns section**: Maps expected column names to actual names in your database. Required fields:
- `label`: The target variable (0/1 for neutral/disease)
- `is_synonymous`: Boolean indicating synonymous mutations
- `pkey`: Primary key column
- `gene`: Gene identifier column

**Features section**: Controls feature detection.
- `auto_detect: true`: Automatically detects numeric columns as features
- `exclude`: Column names to exclude from features

## Usage

### Stage 1: Feature Selection

```bash
# From SQL database
python stage1_feature_selection.py \
    --sql \
    --config config.json \
    --output-dir results/stage1

# From TSV file
python stage1_feature_selection.py \
    --input data.tsv \
    --output-dir results/stage1

# Options
    --cv-folds 5              # Cross-validation folds (default: 5)
    --rf-estimators 500       # Random Forest trees (default: 500)
    --rf-max-depth 6          # Max tree depth (default: 6)
    --elastic-net-l1-ratio 0.5  # Elastic Net mixing (default: 0.5)
```

**Outputs:**
- `importance_table.tsv`: Combined importance scores from all methods
- `l1_selected_features.json`: Features selected by L1
- `feature_selection_results.json`: Full results for Stage 2
- `shap_values.npy`: Raw SHAP values
- `rf_importance.png`, `shap_summary.png`: Visualizations

### Stage 2: Analysis

```bash
python stage2_analysis.py \
    --stage1-results results/stage1 \
    --sql \
    --config config.json \
    --output-dir results/stage2

# Options
    --correlation-threshold 0.7  # For feature clustering (default: 0.7)
    --rf-top-k 20               # Top RF features for intersection (default: 20)
    --max-recommended N         # Max features to recommend (default: all features)
```

**Outputs:**
- `high_confidence_features.json`: Features selected by both L1 and RF
- `ranking_comparison.tsv`: Linear vs non-linear importance comparison
- `feature_clusters.json`: Correlated feature groups
- `mechanism_indices.json`: Feature indices by biological mechanism
- `stratified_importance.json`: Importance by mutation type
- `recommended_features.json`: Features for Stage 3
- `analysis_report.md`: Summary report

### Stage 3: Neural Network

```bash
python stage3_neural_network.py \
    --input results/stage1 \
    --stage2-results results/stage2 \
    --output-dir results/stage3 \
    --architecture base

# Architecture options
    --architecture base       # Standard dense layers
    --architecture attention  # Attention over mechanism groups
    --architecture gated      # Mutation-type conditioned gating

# Training options
    --epochs 100              # Max epochs (default: 100)
    --batch-size 64           # Batch size (default: 64)
    --lr 0.001                # Learning rate (default: 0.001)
    --hidden-dims 64 32       # Hidden layer sizes (default: 64 32)
    --dropout 0.3             # Dropout rate (default: 0.3)
    --cv-folds 10             # Cross-validation folds (default: 10, 0 for single split)
    --cpu                     # Force CPU (default: GPU if available)
```

**Outputs:**
- `cv_results.json`: Cross-validation metrics
- `cv_predictions.npy`: Predicted probabilities
- `pr_curve_cv.png`: Precision-recall curve
- `model.pt`: Trained model weights (single split mode)
- `attention_weights.npy`: Mechanism attention weights (AttentionNN only)

## Architecture Options

### BaseNN
Standard feedforward network. Features pass through dense layers with batch normalization, ReLU activation, and dropout. Learns implicit feature weighting through gradient descent.

### AttentionNN
Groups features by biological mechanism (splicing, RNA structure, miRNA, etc.) and learns attention weights over mechanisms. Provides interpretable output showing which mechanisms drive predictions.

### GatedNN
Uses the `is_synonymous` flag to gate between regulatory features (splicing, RNA, miRNA) and protein features (PTMs, structure). Learns to emphasize different feature types based on mutation type.

## Evaluation Metrics

- **PR-AUC**: Primary metric for imbalanced data
- **MCC**: Matthews Correlation Coefficient
- **F1**: Harmonic mean of precision and recall
- **ECE**: Expected Calibration Error (probability calibration quality)

Metrics are reported for:
- All mutations (joint training performance)
- Synonymous mutations only (target domain)
- Nonsynonymous mutations only (comparison)

## Class Imbalance Handling

The pipeline addresses the ~13:1 negative:positive ratio through:
1. **Class weights**: Positives weighted proportionally higher in loss
2. **Focal Loss**: Down-weights easy examples, focuses on hard positives
3. **Stratified CV**: Maintains class ratio in each fold

## Missing Value Handling

Data loader supports four strategies via `--missing-strategy`:
- `drop`: Remove rows with any missing values
- `mean`: Impute with feature means
- `median`: Impute with feature medians
- `indicator`: Add `feature_missing_count` column (0 to n missing per row), fill with 0

## Example Workflow

```bash
# 1. Feature selection
python stage1_feature_selection.py \
    --sql --config config.json \
    --output-dir results/stage1

# 2. Analyze features
python stage2_analysis.py \
    --stage1-results results/stage1 \
    --sql --config config.json \
    --output-dir results/stage2

# 3. Train neural network (try all architectures)
for arch in base attention gated; do
    python stage3_neural_network.py \
        --input results/stage1 \
        --stage2-results results/stage2 \
        --output-dir results/stage3_${arch} \
        --architecture ${arch} \
        --cv-folds 10
done
```

## Documentation

See `stage_explanations.txt` for detailed pseudocode explanations of each stage's implementation.
