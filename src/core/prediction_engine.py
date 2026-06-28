"""Prediction engine for IDS"""

import logging
import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PredictionEngine:
    """Real-time prediction engine"""
    
    def __init__(self, model_path, scaler_path=None):
        self.model = joblib.load(model_path)
        self.scaler = None
        
        if scaler_path:
            self.scaler = joblib.load(scaler_path)
        
        logger.info(f"✅ Prediction engine loaded from {model_path}")
    
    def predict(self, X):
        """Make predictions"""
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict_proba(X_scaled)
    
    def predict_with_confidence(self, X, threshold=0.5):
        """Predict with confidence scores"""
        y_proba = self.predict_proba(X)
        y_pred = (y_proba[:, 1] >= threshold).astype(int)
        
        return {
            'predictions': y_pred,
            'probabilities': y_proba[:, 1],
            'confidence': np.max(y_proba, axis=1)
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Prediction engine module loaded")
