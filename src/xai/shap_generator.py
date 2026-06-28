"""
SHAP Value Generation and Analysis

Generate SHAP values and save them to CSV for feature importance analysis.
"""

import pandas as pd
import numpy as np
import joblib
import shap
import os


def generate_shap_csv(model, X_data, output_path='results/xai_explanations/shap_values.csv', 
                      model_type='tree'):
    """
    Generate SHAP values and save to CSV.
    
    Args:
        model: Trained model
        X_data: Feature data (pandas DataFrame)
        output_path: Path to save SHAP values CSV
        model_type: Type of model ('tree', 'linear', 'kernel')
        
    Returns:
        DataFrame with SHAP values
    """
    print("🔍 Generating SHAP values...")
    
    # Create explainer
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
    elif model_type == 'linear':
        explainer = shap.LinearExplainer(model, X_data)
    else:
        explainer = shap.KernelExplainer(model.predict, X_data)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_data)
    
    # Handle binary classification
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            shap_values = shap_values[1]  # Attack class
        else:
            raise ValueError("Unexpected number of classes from SHAP")
    
    # Create DataFrame with SHAP values
    shap_df = pd.DataFrame(
        shap_values,
        columns=X_data.columns
    )
    
    # Compute mean absolute SHAP values for feature importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        'feature': X_data.columns,
        'mean_abs_shap': mean_abs_shap,
        'mean_shap': shap_values.mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    
    # Save both DataFrames
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shap_df.to_csv(output_path, index=False)
    
    importance_path = output_path.replace('.csv', '_importance.csv')
    feature_importance_df.to_csv(importance_path, index=False)
    
    print(f"✅ SHAP values saved to: {output_path}")
    print(f"✅ Feature importance saved to: {importance_path}")
    
    return feature_importance_df


def get_top_features_from_shap(shap_csv_path, top_n=10):
    """
    Get top N features from SHAP importance CSV.
    
    Args:
        shap_csv_path: Path to SHAP importance CSV
        top_n: Number of top features to return
        
    Returns:
        List of top feature names
    """
    shap_df = pd.read_csv(shap_csv_path)
    return shap_df['feature'].head(top_n).tolist()


def analyze_shap_values(shap_values_df, feature_importance_df):
    """
    Analyze SHAP values and print summary statistics.
    
    Args:
        shap_values_df: DataFrame of SHAP values
        feature_importance_df: DataFrame of feature importance
    """
    print("\n📊 SHAP Analysis Summary")
    print("=" * 60)
    print(f"\nTotal Features: {len(shap_values_df.columns)}")
    print(f"Total Instances: {len(shap_values_df)}")
    
    print("\n🔝 Top 10 Most Important Features (by mean absolute SHAP):")
    print(feature_importance_df.head(10).to_string(index=False))
    
    print("\n📈 SHAP Value Statistics:")
    print(f"Mean SHAP value: {shap_values_df.values.mean():.4f}")
    print(f"Std SHAP value: {shap_values_df.values.std():.4f}")
    print(f"Max SHAP value: {shap_values_df.values.max():.4f}")
    print(f"Min SHAP value: {shap_values_df.values.min():.4f}")
