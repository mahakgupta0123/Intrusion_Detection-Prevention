"""
Model Training Module

Handles training of various IDS models (Random Forest, XGBoost, etc.)
"""

import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Unified model training interface"""
    
    def __init__(self, model_dir='trained_models', random_state=42):
        self.model_dir = model_dir
        self.random_state = random_state
        self.model = None
        os.makedirs(model_dir, exist_ok=True)
    
    def train_random_forest(self, X_train, y_train, n_estimators=200, max_depth=15):
        """Train Random Forest classifier"""
        logger.info(f"🌲 Training Random Forest (n_estimators={n_estimators})...")
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        logger.info("✅ Random Forest training complete")
        
        return self.model
    
    def train_xgboost(self, X_train, y_train, n_estimators=200, max_depth=6, learning_rate=0.05):
        """Train XGBoost classifier with proper class weighting"""
        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.error("XGBoost not installed. Install with: pip install xgboost")
            return None
        
        logger.info(f"🚀 Training XGBoost (n_estimators={n_estimators})...")
        
        # Calculate class weight
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        logger.info(f"   Minority/majority weight ratio: {scale_pos_weight:.2f}")
        
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=self.random_state,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        logger.info("✅ XGBoost training complete")
        
        return self.model
    
    def evaluate(self, X_test, y_test, model=None):
        """Evaluate model on test set"""
        if model is None:
            model = self.model
        
        if model is None:
            raise ValueError("No model trained yet")
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        logger.info("\n📊 Test Set Metrics:")
        for metric, value in metrics.items():
            logger.info(f"   {metric}: {value:.4f}")
        
        return metrics, y_pred, y_proba
    
    def save(self, name='model.pkl'):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        path = os.path.join(self.model_dir, name)
        joblib.dump(self.model, path)
        logger.info(f"✅ Model saved to {path}")
    
    def load(self, name='model.pkl'):
        """Load trained model"""
        path = os.path.join(self.model_dir, name)
        self.model = joblib.load(path)
        logger.info(f"✅ Model loaded from {path}")
        return self.model


def train_random_forest(X_train, y_train, X_test=None, y_test=None, **kwargs):
    """Train Random Forest (standalone function)"""
    trainer = ModelTrainer()
    trainer.train_random_forest(X_train, y_train, **kwargs)
    
    if X_test is not None and y_test is not None:
        metrics, y_pred, y_proba = trainer.evaluate(X_test, y_test)
        return trainer.model, metrics
    
    return trainer.model


def train_xgboost(X_train, y_train, X_test=None, y_test=None, **kwargs):
    """Train XGBoost (standalone function)"""
    trainer = ModelTrainer()
    trainer.train_xgboost(X_train, y_train, **kwargs)
    
    if X_test is not None and y_test is not None:
        metrics, y_pred, y_proba = trainer.evaluate(X_test, y_test)
        return trainer.model, metrics
    
    return trainer.model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Model trainer module loaded")
