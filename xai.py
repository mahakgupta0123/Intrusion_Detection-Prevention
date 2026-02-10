import os
import shap
import lime
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lime.lime_tabular

# Paths
MODEL_PATH = 'trained_models/xgb_model_unsw.pkl'
X_TEST_PATH = 'data/processed/X_test_unsw.csv'
Y_TEST_PATH = 'data/processed/y_test_unsw.csv'
RESULTS_DIR = 'results/xai_explanations'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load data and model
def load_model_and_data():
    model = joblib.load(MODEL_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()
    return model, X_test, y_test

# SHAP Explainability
def run_shap_explanation(model, X_test):
    print("🔍 Running SHAP explainability...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Bar plot (for top contributing features)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Bar)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'shap_bar_plot.png'))
    plt.close()

    # Beeswarm plot
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary Plot (Beeswarm)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'shap_beeswarm_plot.png'))
    plt.close()

    # Force plot as HTML (not matplotlib image)
    shap.initjs()
    force_plot = shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
    shap.save_html(os.path.join(RESULTS_DIR, "shap_force_plot_instance.html"), force_plot)

    print("✅ SHAP plots saved: bar, beeswarm, and force (HTML)")

# LIME Explainability
def run_lime_explanation(model, X_test):
    print("🔍 Running LIME explainability...")
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_test),
        feature_names=X_test.columns.tolist(),
        class_names=['Normal', 'Attack'],
        mode='classification',
        verbose=True,
        random_state=42
    )

    explanation = explainer.explain_instance(
        data_row=X_test.iloc[0],
        predict_fn=model.predict_proba
    )

    explanation.save_to_file(os.path.join(RESULTS_DIR, "lime_explanation_instance.html"))
    print("✅ LIME explanation saved (HTML format).")

# Run all
if __name__ == "__main__":
    model, X_test, y_test = load_model_and_data()
    run_shap_explanation(model, X_test)
    run_lime_explanation(model, X_test)
    print("🎯 XAI (SHAP + LIME) completed successfully.")
