"""
XAI (Explainable AI) Module

This module provides explainability capabilities using SHAP and LIME
for understanding model predictions.

Classes:
    SHAPExplainer - SHAP-based model explainability
    LIMEExplainer - LIME-based model explainability

Functions:
    run_shap_explanation() - Generate SHAP explanations
    run_lime_explanation() - Generate LIME explanations
    generate_shap_csv() - Generate SHAP values to CSV

Usage:
    from src.xai import SHAPExplainer
    
    explainer = SHAPExplainer(model, X_test)
    explainer.run_shap_explanation()
    explainer.save_plots('results/xai/')
"""

from .xai_explainer import (
    SHAPExplainer,
    LIMEExplainer,
    run_shap_explanation,
    run_lime_explanation
)

from .shap_generator import (
    generate_shap_csv,
    get_top_features_from_shap
)

__all__ = [
    'SHAPExplainer',
    'LIMEExplainer',
    'run_shap_explanation',
    'run_lime_explanation',
    'generate_shap_csv',
    'get_top_features_from_shap'
]
