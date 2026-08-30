#!/usr/bin/env python3
"""
Runs sequentially:
load → validate → preprocess → feature engineering
→ train → evaluate → log to MLflow
"""

import os
import sys
import time
import argparse
import json
import joblib

# Windows terminals often default to cp1252, which cannot print the status
# symbols used by this script.  Make command-line execution deterministic.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import mlflow
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from xgboost import XGBClassifier


# ============================================================
# Fix import path for local modules
# ============================================================
# Allows imports from the src/ directory structure
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)


# ============================================================
# Local modules - Core pipeline components
# ============================================================

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_telco_data


# ============================================================
# Main training pipeline
# ============================================================

def main(args):
    """
    Main training pipeline function.

    Pipeline:
        1. Load data
        2. Validate data
        3. Preprocess data
        4. Build features
        5. Split train/test
        6. Handle class imbalance
        7. Train XGBoost
        8. Evaluate model
        9. Save model and preprocessing artifacts to MLflow
    """

    # ========================================================
    # MLflow Setup
    # ========================================================

    # Get project root directory
    project_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )

    # MLflow 3 no longer enables the file-store backend by default.  A local
    # SQLite database remains self-contained while working with current MLflow.
    mlruns_path = args.mlflow_uri or f"sqlite:///{project_root.replace(os.sep, '/')}/mlflow.db"

    # Configure MLflow tracking URI
    mlflow.set_tracking_uri(mlruns_path)

    # Create/use MLflow experiment
    mlflow.set_experiment(args.experiment)


    # ========================================================
    # Start MLflow Run
    # ========================================================

    with mlflow.start_run():

        # ====================================================
        # Log configuration parameters
        # ====================================================

        mlflow.log_param("model", "xgboost")
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("test_size", args.test_size)


        # ====================================================
        # STAGE 1: Data Loading
        # ====================================================

        print("🔄 Loading data...")

        df = load_data(args.input)

        print(
            f"✅ Data loaded: "
            
            f"{df.shape[0]} rows, "
            f"{df.shape[1]} columns"
        )


        # ====================================================
        # STAGE 1.5: Data Validation
        # ====================================================

        print(
            "🔍 Validating data quality..."
        )

        is_valid, failed = validate_telco_data(df)

        # Save validation result to MLflow
        mlflow.log_metric(
            "data_quality_pass",
            int(is_valid)
        )


        # If validation failed, stop pipeline
        if not is_valid:

            mlflow.log_text(
                json.dumps(
                    failed,
                    indent=2
                ),
                artifact_file="failed_expectations.json"
            )

            raise ValueError(
                f"❌ Data quality check failed. "
                f"Issues: {failed}"
            )

        else:

            print(
                "✅ Data validation passed. "
                "Logged to MLflow."
            )


        # ====================================================
        # STAGE 2: Data Preprocessing
        # ====================================================

        print("🔧 Preprocessing data...")

        df = preprocess_data(df)


        # ====================================================
        # Save processed dataset
        # ====================================================

        processed_path = os.path.join(
            project_root,
            "data",
            "processed",
            "telco_churn_processed.csv"
        )

        # Create directory if it doesn't exist
        os.makedirs(
            os.path.dirname(processed_path),
            exist_ok=True
        )

        # Save processed dataset
        df.to_csv(
            processed_path,
            index=False
        )

        print(
            f"✅ Processed dataset saved to "
            f"{processed_path} | "
            f"Shape: {df.shape}"
        )


        # ====================================================
        # STAGE 3: Feature Engineering
        # ====================================================

        print("🛠️ Building features...")

        target = args.target


        # Check that target column exists
        if target not in df.columns:

            raise ValueError(
                f"Target column '{target}' "
                f"not found in data"
            )


        # Apply feature engineering
        # Example:
        # binary encoding + one-hot encoding
        df_enc = build_features(
            df,
            target_col=target
        )


        # ====================================================
        # Convert boolean columns to integers
        # ====================================================

        for column in df_enc.select_dtypes(
            include=["bool"]
        ).columns:

            df_enc[column] = (
                df_enc[column].astype(int)
            )


        print(
            f"✅ Feature engineering completed: "
            f"{df_enc.shape[1]} features"
        )


        # ====================================================
        # STAGE 3.5: Save Feature Metadata
        # ====================================================

        artifacts_dir = os.path.join(
            project_root,
            "artifacts"
        )

        # Create artifacts directory
        os.makedirs(
            artifacts_dir,
            exist_ok=True
        )


        # Get feature columns
        # Exclude target column
        feature_cols = list(
            df_enc
            .drop(columns=[target])
            .columns
        )


        # ====================================================
        # Save feature columns as JSON
        # ====================================================

        feature_columns_path = os.path.join(
            artifacts_dir,
            "feature_columns.json"
        )

        with open(
            feature_columns_path,
            "w"
        ) as f:

            json.dump(
                feature_cols,
                f,
                indent=2
            )


        # ====================================================
        # Log feature columns to MLflow
        # ====================================================

        mlflow.log_text(
            "\n".join(feature_cols),
            artifact_file="feature_columns.txt"
        )


        # ====================================================
        # Save preprocessing metadata
        # ====================================================

        preprocessing_artifact = {

            # Exact feature order
            "feature_columns": feature_cols,

            # Target column
            "target": target
        }


        preprocessing_path = os.path.join(
            artifacts_dir,
            "preprocessing.pkl"
        )


        joblib.dump(
            preprocessing_artifact,
            preprocessing_path
        )


        # Log preprocessing artifact to MLflow
        mlflow.log_artifact(
            preprocessing_path
        )


        print(
            f"✅ Saved {len(feature_cols)} "
            f"feature columns for serving consistency"
        )


        # ====================================================
        # STAGE 4: Train/Test Split
        # ====================================================

        print("📊 Splitting data...")


        # Feature matrix
        X = df_enc.drop(
            columns=[target]
        )


        # Target vector
        y = df_enc[target]


        # Split data
        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,

            # 20% test by default
            test_size=args.test_size,

            # Keep same class distribution
            stratify=y,

            # Reproducibility
            random_state=42
        )


        print(
            f"✅ Train: {X_train.shape[0]} samples | "
            f"Test: {X_test.shape[0]} samples"
        )


        # ====================================================
        # STAGE 4.5: Handle Class Imbalance
        # ====================================================

        # Number of negative samples / number of positive samples
        #
        # 0 = no churn
        # 1 = churn

        scale_pos_weight = (
            (y_train == 0).sum()
            /
            (y_train == 1).sum()
        )


        print(
            f"📈 Class imbalance ratio: "
            f"{scale_pos_weight:.2f} "
            f"(applied to positive class)"
        )


        # Log class imbalance ratio to MLflow
        mlflow.log_param(
            "scale_pos_weight",
            scale_pos_weight
        )


        # ====================================================
        # STAGE 5: XGBoost Model
        # ====================================================

        print("🤖 Training XGBoost model...")


        model = XGBClassifier(

            # Number of trees
            n_estimators=301,

            # Learning rate
            learning_rate=0.034,

            # Maximum tree depth
            max_depth=7,

            # Percentage of training samples
            # used for each tree
            subsample=0.95,

            # Percentage of features
            # used for each tree
            colsample_bytree=0.98,

            # Use all CPU cores
            n_jobs=-1,

            # Reproducibility
            random_state=42,

            # Training evaluation metric
            eval_metric="logloss",

            # Handle class imbalance
            scale_pos_weight=scale_pos_weight
        )


        # ====================================================
        # Train Model
        # ====================================================

        t0 = time.time()


        model.fit(
            X_train,
            y_train
        )


        train_time = time.time() - t0


        # Log training time
        mlflow.log_metric(
            "train_time",
            train_time
        )


        print(
            f"✅ Model trained in "
            f"{train_time:.2f} seconds"
        )


        # ====================================================
        # STAGE 6: Model Evaluation
        # ====================================================

        print("📊 Evaluating model performance...")


        # Start inference timer
        t1 = time.time()


        # Get probability of class 1 = churn
        proba = model.predict_proba(
            X_test
        )[:, 1]


        # Apply classification threshold
        #
        # Example:
        #
        # probability >= 0.35 → churn
        # probability < 0.35  → no churn

        y_pred = (
            proba >= args.threshold
        ).astype(int)


        # Calculate inference time
        pred_time = time.time() - t1


        # Log inference time
        mlflow.log_metric(
            "pred_time",
            pred_time
        )


        # ====================================================
        # Calculate Metrics
        # ====================================================

        precision = precision_score(
            y_test,
            y_pred
        )


        recall = recall_score(
            y_test,
            y_pred
        )


        f1 = f1_score(
            y_test,
            y_pred
        )


        # ROC-AUC uses probabilities,
        # not thresholded predictions
        roc_auc = roc_auc_score(
            y_test,
            proba
        )


        # ====================================================
        # Log Metrics to MLflow
        # ====================================================

        mlflow.log_metric(
            "precision",
            precision
        )


        mlflow.log_metric(
            "recall",
            recall
        )


        mlflow.log_metric(
            "f1",
            f1
        )


        mlflow.log_metric(
            "roc_auc",
            roc_auc
        )


        # ====================================================
        # Display Performance
        # ====================================================

        print("\n🎯 Model Performance:")

        print(
            f"   Precision: {precision:.3f}"
        )

        print(
            f"   Recall:    {recall:.3f}"
        )

        print(
            f"   F1 Score:  {f1:.3f}"
        )

        print(
            f"   ROC AUC:   {roc_auc:.3f}"
        )


        # ====================================================
        # STAGE 7: Save Model to MLflow
        # ====================================================

        print("💾 Saving model to MLflow...")


        # Keep a stable local copy for the FastAPI/Gradio serving application.
        model_path = os.path.join(artifacts_dir, "model.json")
        model.save_model(model_path)
        mlflow.log_artifact(model_path)

        # XGBoost has its own MLflow flavour.  Logging it as an sklearn model
        # fails in MLflow 3 because the booster is intentionally not skops-safe.
        mlflow.xgboost.log_model(model, name="model")


        print(
            "✅ Model saved to MLflow "
            "for serving pipeline"
        )


        # ====================================================
        # Final Performance Summary
        # ====================================================

        print("\n⏱️ Performance Summary:")

        print(
            f"   Training time: "
            f"{train_time:.2f}s"
        )

        print(
            f"   Inference time: "
            f"{pred_time:.4f}s"
        )


        # Avoid division by zero
        if pred_time > 0:

            print(
                f"   Samples per second: "
                f"{len(X_test) / pred_time:.0f}"
            )


        # ====================================================
        # Classification Report
        # ====================================================

        print(
            "\n📈 Detailed Classification Report:"
        )

        print(
            classification_report(
                y_test,
                y_pred,
                digits=3
            )
        )


# ============================================================
# Command Line Interface
# ============================================================

if __name__ == "__main__":

    # Create argument parser
    p = argparse.ArgumentParser(
        description=(
            "Run churn pipeline "
            "with XGBoost + MLflow"
        )
    )


    # Input CSV
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Path to CSV "
            "(e.g., data/raw/Telco-Customer-Churn.csv)"
        )
    )


    # Target column
    p.add_argument(
        "--target",
        type=str,
        default="Churn",
        help="Target column name"
    )


    # Classification threshold
    p.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Classification threshold"
    )


    # Test size
    p.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test set ratio"
    )


    # MLflow experiment
    p.add_argument(
        "--experiment",
        type=str,
        default="Telco Churn",
        help="MLflow experiment name"
    )


    # MLflow tracking URI
    p.add_argument(
        "--mlflow_uri",
        type=str,
        default=None,
        help=(
            "Override MLflow tracking URI. "
            "Otherwise uses project_root/mlruns"
        )
    )


    # Parse command-line arguments
    args = p.parse_args()


    # Run pipeline
    main(args)


"""
============================================================
Example:
============================================================

Run the pipeline from the project root:

python scripts/run_pipeline.py \
    --input data/raw/Telco-Customer-Churn.csv \
    --target Churn


Using uv:

uv run python scripts/run_pipeline.py \
    --input data/raw/Telco-Customer-Churn.csv \
    --target Churn


Change threshold:

uv run python scripts/run_pipeline.py \
    --input data/raw/Telco-Customer-Churn.csv \
    --target Churn \
    --threshold 0.30
"""

