"""
XAI Explainability Module

Provides SHAP and LIME explainability for intrusion detection models.
"""

import os
import shap
import lime
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lime.lime_tabular
from pathlib import Path


class SHAPExplainer:
    """SHAP-based model explainability."""
    
    def __init__(self, model, X_test, model_type='tree'):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model (XGBoost, RandomForest, etc.)
            X_test: Test features (pandas DataFrame)
            model_type: Type of model ('tree', 'linear', 'kernel')
        """
        self.model = model
        self.X_test = X_test
        self.model_type = model_type
        
        # Create explainer based on model type
        if model_type == 'tree':
            self.explainer = shap.TreeExplainer(model)
        elif model_type == 'linear':
            self.explainer = shap.LinearExplainer(model, X_test)
        else:
            self.explainer = shap.KernelExplainer(model.predict, X_test)
        
        self.shap_values = None
        self.expected_value = None
    
    def compute_shap_values(self):
        """Compute SHAP values for test data."""
        print("🔍 Computing SHAP values...")
        self.shap_values = self.explainer.shap_values(self.X_test)
        
        if isinstance(self.shap_values, list):
            self.expected_value = self.explainer.expected_value
            if isinstance(self.expected_value, list):
                self.expected_value = self.expected_value[1]  # Attack class
            self.shap_values = self.shap_values[1]  # Attack class
        else:
            self.expected_value = self.explainer.expected_value
        
        print("✅ SHAP values computed")
        return self.shap_values
    
    def save_summary_plot(self, output_dir):
        """Save SHAP summary plot (beeswarm)."""
        if self.shap_values is None:
            self.compute_shap_values()
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("📊 Saving SHAP summary plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(self.shap_values, self.X_test, show=False)
        plt.title("SHAP Summary Plot (Beeswarm)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_summary_beeswarm.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {os.path.join(output_dir, 'shap_summary_beeswarm.png')}")
    
    def save_bar_plot(self, output_dir):
        """Save SHAP bar plot (mean absolute values)."""
        if self.shap_values is None:
            self.compute_shap_values()
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("📊 Saving SHAP bar plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(self.shap_values, self.X_test, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance (Bar)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_bar_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {os.path.join(output_dir, 'shap_bar_plot.png')}")
    
    def save_force_plot(self, output_dir, instance_idx=0):
        """Save SHAP force plot as HTML."""
        if self.shap_values is None:
            self.compute_shap_values()
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📊 Saving SHAP force plot for instance {instance_idx}...")
        shap.initjs()
        force_plot = shap.force_plot(
            self.expected_value,
            self.shap_values[instance_idx],
            self.X_test.iloc[instance_idx]
        )
        shap.save_html(os.path.join(output_dir, f"shap_force_plot_instance_{instance_idx}.html"), force_plot)
        print(f"✅ Saved: {os.path.join(output_dir, f'shap_force_plot_instance_{instance_idx}.html')}")
    
    def save_all_plots(self, output_dir):
        """Save all SHAP plots."""
        print(f"📈 Saving all SHAP plots to {output_dir}...")
        self.save_bar_plot(output_dir)
        self.save_summary_plot(output_dir)
        self.save_force_plot(output_dir)
        print("✅ All SHAP plots saved")
    
    def get_feature_importance(self):
        """Get top important features based on SHAP values."""
        if self.shap_values is None:
            self.compute_shap_values()
        
        # Mean absolute SHAP values
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': self.X_test.columns,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        return feature_importance


class LIMEExplainer:
    """LIME-based model explainability."""
    
    def __init__(self, model, X_train, feature_names=None):
        """
        Initialize LIME explainer.
        
        Args:
            model: Trained model (any sklearn-compatible model)
            X_train: Training features (pandas DataFrame or array)
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train.values if hasattr(X_train, 'values') else X_train
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(self.X_train.shape[1])]
        
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train=self.X_train,
            feature_names=feature_names,
            class_names=['Normal', 'Attack'],
            mode='classification'
        )
    
    def explain_instance(self, instance, num_features=10):
        """
        Explain a single instance.
        
        Args:
            instance: Feature vector to explain
            num_features: Number of features to show
            
        Returns:
            LIME explanation object
        """
        instance = instance.values if hasattr(instance, 'values') else instance
        explanation = self.explainer.explain_instance(
            instance,
            self.model.predict_proba,
            num_features=num_features
        )
        return explanation
    
    def save_explanation_html(self, instance, output_path):
        """Save LIME explanation as HTML."""
        explanation = self.explain_instance(instance)
        explanation.save_to_file(output_path)
        print(f"✅ Saved LIME explanation: {output_path}")


def run_shap_explanation(model, X_test, output_dir='results/xai_explanations'):
    """
    Run complete SHAP explainability pipeline.
    
    Args:
        model: Trained model
        X_test: Test features (pandas DataFrame)
        output_dir: Output directory for plots
        
    Returns:
        SHAPExplainer instance
    """
    explainer = SHAPExplainer(model, X_test)
    explainer.save_all_plots(output_dir)
    return explainer


def run_lime_explanation(model, X_train, X_test, instance_idx=0, output_dir='results/xai_explanations'):
    """
    Run complete LIME explainability for a single instance.
    
    Args:
        model: Trained model
        X_train: Training features
        X_test: Test features
        instance_idx: Instance to explain
        output_dir: Output directory for HTML
        
    Returns:
        LIME explanation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    explainer = LIMEExplainer(model, X_train, feature_names=X_test.columns.tolist())
    instance = X_test.iloc[instance_idx]
    explanation = explainer.explain_instance(instance)
    
    output_path = os.path.join(output_dir, f'lime_explanation_instance_{instance_idx}.html')
    explainer.save_explanation_html(instance, output_path)
    
    return explanation
