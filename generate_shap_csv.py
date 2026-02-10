import pandas as pd
import joblib
import shap

# === Paths ===
MODEL_PATH = 'trained_models/prevention_model_unsw.pkl'
DATA_PATH = r'C:\IPDRS_research\data\UNSW‑NB15\UNSW_NB15_training-set.csv'
OUTPUT_PATH = 'results/xai_explanations/shap_values_unsw.csv'

# === Load data and model ===
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# Use only model input features
model_features = model.feature_names_in_
X = df[model_features]

# === Create SHAP explainer and values ===
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Handle binary classification and extract attack class SHAP
if isinstance(shap_values, list):  # old style or multiclass
    if len(shap_values) == 2:
        contributions = shap_values[1]  # Class 1
    else:
        raise ValueError("SHAP returned unexpected number of classes.")
else:
    contributions = shap_values

# === Calculate mean absolute SHAP for each feature ===
mean_abs_shap = pd.DataFrame({
    'feature': model_features,
    'mean_abs_shap': abs(contributions).mean(axis=0)
}).sort_values(by='mean_abs_shap', ascending=False)

# === Save to CSV ===
mean_abs_shap.to_csv(OUTPUT_PATH, index=False)
print(f"✅ SHAP values saved to: {OUTPUT_PATH}")
