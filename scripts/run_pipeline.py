#!/usr/bin/env python3
"""
Runs sequentially: load → validate → preprocess → feature engineering
"""

import os
import sys
import time
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from posthog import project_root
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# === Fix import path for local modules ===
# ESSENTIAL: Allows imports from src/ directory structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local modules - Core pipeline components
from src.data.load_data import load_data  # Data loading with error handling
from src.data.preprocess import preprocess_data  # Basic data cleaning
from src.features.build_features import (
    build_features,
)  # Feature engineering (CRITICAL for model performance)
from src.utils.validate_data import validate_telco_data  # Data quality validation


def main(args):
    """
    Main training pipeline function that orchestrates the complete ML workflow.

    """


    # === MLflow Setup - ESSENTIAL for experiment tracking ===
    # Configure MLflow to use local file-based tracking (not a tracking server)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
    mlruns_path = args.mlflow_url or f"file://{project_root}/mlruns"  # LOcal file based tracking 
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)  # creats experiment if doesn't exist
    
    #start mlflow run - all subsequent will be tracked under this run 
    with mlflow.start_run():
        # === Log hyperparameters and configuration ===
        # REQUIRED: These parameters are essential for model reproducibility
        mlflow.log_param("model", "xgboost")           # Model type for comparison
        mlflow.log_param("threshold", args.threshold)   # Classification threshold (default: 0.35)
        mlflow.log_param("test_size", args.test_size)   # Train/test split ratio
        
        # === STAGE 1: Data Loading & Validation ===
        print("🔄 Loading data...")
        df = load_data(args.input)  # Load raw CSV data with error handling
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")