#!/usr/bin/env python3
"""
Model Evaluation Script

Evaluates models with stratified k-fold cross-validation.
"""

import sys
import logging
import os
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation import stratified_kfold_evaluation, compare_models_with_cv


if __name__ == "__main__":
    data_dir = 'data/processed'
    
    logger.info("📥 Loading data...")
    
    try:
        X_train = pd.read_csv(os.path.join(data_dir, 'X_train_unsw_augmented.csv'))
        y_train = pd.read_csv(os.path.join(data_dir, 'y_train_unsw_augmented.csv')).squeeze()
    except FileNotFoundError:
        logger.warning("Augmented data not found, using original...")
        X_train = pd.read_csv(os.path.join(data_dir, 'X_train_unsw.csv'))
        y_train = pd.read_csv(os.path.join(data_dir, 'y_train_unsw.csv')).squeeze()
    
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test_unsw.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test_unsw.csv')).squeeze()
    
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Evaluate individual models
    for model_type in ['rf', 'xgb']:
        logger.info(f"\n\n{'#'*80}\n# Evaluating {model_type.upper()}\n{'#'*80}")
        try:
            results = stratified_kfold_evaluation(
                X_train, y_train, X_test, y_test,
                model_type=model_type,
                n_splits=5
            )
        except Exception as e:
            logger.error(f"Evaluation failed for {model_type}: {e}")
    
    # Compare models
    logger.info("\n\n" + "="*80)
    logger.info("📊 Comparing All Models")
    logger.info("="*80)
    
    try:
        df_comparison = compare_models_with_cv(X_train, y_train, X_test, y_test)
    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
    
    logger.info("✅ Evaluation complete!")
