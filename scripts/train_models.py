#!/usr/bin/env python3
"""
Model Training Script

Trains different model types with augmented data.
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

from src.models import ModelTrainer


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
    
    trainer = ModelTrainer(model_dir='trained_models')
    
    # Train Random Forest
    logger.info("\n🌲 Training Random Forest...")
    try:
        trainer.train_random_forest(X_train, y_train)
        trainer.evaluate(X_test, y_test)
        trainer.save('ids_model_rf_augmented.pkl')
    except Exception as e:
        logger.error(f"RF training failed: {e}")
    
    # Train XGBoost
    logger.info("\n🚀 Training XGBoost...")
    try:
        trainer.train_xgboost(X_train, y_train)
        trainer.evaluate(X_test, y_test)
        trainer.save('ids_model_xgb_augmented.pkl')
    except Exception as e:
        logger.error(f"XGBoost training failed: {e}")
    
    logger.info("✅ Model training complete!")
