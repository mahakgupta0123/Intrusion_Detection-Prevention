"""
Prevention Model Training

Train and deploy prevention models for attack mitigation.
"""

import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from datetime import datetime


class PreventionModel:
    """Train and deploy prevention models."""
    
    def __init__(self, model_type='xgboost'):
        """
        Initialize prevention model.
        
        Args:
            model_type: Type of model ('xgboost', 'random_forest')
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.top_features = None
    
    def train(self, X, y, top_features=None, **kwargs):
        """
        Train prevention model.
        
        Args:
            X: Training features (pandas DataFrame)
            y: Training labels
            top_features: List of features to use (if None, use all)
            **kwargs: Additional arguments for model
            
        Returns:
            Trained model
        """
        if top_features is not None:
            X_train = X[top_features]
            self.top_features = top_features
        else:
            X_train = X
            self.top_features = X.columns.tolist()
        
        self.feature_names = self.top_features
        
        print(f"🧠 Training {self.model_type} prevention model...")
        
        if self.model_type == 'xgboost':
            self.model = XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                **kwargs
            )
        else:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(**kwargs)
        
        self.model.fit(X_train, y)
        print("✅ Prevention model trained")
        return self.model
    
    def predict(self, X):
        """
        Make prevention predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Predictions (0=Allow, 1=Block)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.top_features:
            X = X[self.top_features]
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            Probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.top_features:
            X = X[self.top_features]
        
        return self.model.predict_proba(X)
    
    def save(self, save_path):
        """
        Save model to disk.
        
        Args:
            save_path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(self.model, save_path)
        print(f"✅ Model saved to: {save_path}")
    
    def load(self, load_path):
        """
        Load model from disk.
        
        Args:
            load_path: Path to load model
        """
        self.model = joblib.load(load_path)
        print(f"✅ Model loaded from: {load_path}")
    
    def get_feature_importance(self):
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            return feature_importance_df
        else:
            raise ValueError("Model does not support feature importance.")


def train_prevention_model(X, y, top_features, save_path='trained_models/prevention_model.pkl'):
    """
    Train and save prevention model.
    
    Args:
        X: Training features
        y: Training labels
        top_features: Features to use
        save_path: Path to save model
        
    Returns:
        Trained model
    """
    model = PreventionModel(model_type='xgboost')
    model.train(X, y, top_features=top_features)
    model.save(save_path)
    return model.model


def apply_prevention_rules(df, top_features, rules_dict=None):
    """
    Apply prevention rules to data.
    
    Args:
        df: Data to apply rules to
        top_features: Top important features
        rules_dict: Custom rule definitions (optional)
        
    Returns:
        DataFrame with prevent_flag column
    """
    print("⚙️  Applying prevention rules...")
    
    # Default rules based on feature importance
    if rules_dict is None:
        rules_dict = {
            top_features[0]: lambda x: x > 0.7,
            top_features[1]: lambda x: x < -0.5,
            top_features[2]: lambda x: x > 1.2,
            top_features[3]: lambda x: x < 0.3,
            top_features[4]: lambda x: x > 0.8,
        }
    
    df = df.copy()
    df['prevent_flag'] = 0
    
    for feat, rule in rules_dict.items():
        if feat in df.columns:
            df['prevent_flag'] |= df[feat].apply(rule).astype(int)
    
    print(f"✅ Prevention rules applied. {df['prevent_flag'].sum()} packets flagged")
    return df


def get_top_features(shap_csv_path, top_n=5):
    """
    Get top N features from SHAP importance CSV.
    
    Args:
        shap_csv_path: Path to SHAP importance CSV
        top_n: Number of top features to return
        
    Returns:
        List of top feature names
    """
    shap_df = pd.read_csv(shap_csv_path)
    top_features = shap_df['feature'].head(top_n).tolist()
    print(f"📌 Top {top_n} features: {top_features}")
    return top_features
