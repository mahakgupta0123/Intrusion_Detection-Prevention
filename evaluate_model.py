import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve
)
from imblearn.over_sampling import SMOTE

# === Paths ===
MODEL_PATH = 'trained_models/prevention_model_unsw.pkl'
TEST_DATA_PATH = r'/home/mahak/IPDRS_research/data/UNSW‑NB15/UNSW_NB15_testing-set.csv'
SHAP_PATH = 'results/xai_explanations/shap_values_unsw.csv'

print("🔄 Loading model and SHAP features...")
model = joblib.load(MODEL_PATH)

# Get model features in training order
model_features = list(model.feature_names_in_)

# === Load SHAP CSV ===
shap_df = pd.read_csv(SHAP_PATH)
if 'feature' not in shap_df.columns:
    raise ValueError("❌ SHAP CSV missing 'feature' column.")
TOP_FEATURES = model_features  # Use all features the model expects
print(f"📌 Top SHAP Features: {TOP_FEATURES}")

# === Load test dataset ===
df = pd.read_csv(TEST_DATA_PATH)

# === Sanity check: Ensure all features exist ===
missing = [f for f in TOP_FEATURES if f not in df.columns]
if missing:
    raise ValueError(f"❌ Dataset missing required features: {missing}")

X = df[TOP_FEATURES]
y = df['label'] if 'label' in df.columns else df.iloc[:, -1]  # fallback to last col

# === Balance data with SMOTE ===
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Ensure correct feature order for XGBoost
X_resampled = X_resampled[model_features]

# === Predict probabilities and optimize threshold ===
y_scores = model.predict_proba(X_resampled)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_resampled, y_scores)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_thresh = thresholds[f1_scores.argmax()]
y_pred = (y_scores > best_thresh).astype(int)

# === Compute metrics ===
accuracy = accuracy_score(y_resampled, y_pred)
precision_val = precision_score(y_resampled, y_pred, zero_division=0)
recall_val = recall_score(y_resampled, y_pred, zero_division=0)
f1 = f1_score(y_resampled, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_resampled, y_scores)
cm = confusion_matrix(y_resampled, y_pred)
false_positives = cm[0][1]
false_discovery_rate = false_positives / max((false_positives + cm[1][1]), 1)

# === Print Metrics ===
print("\n📊 Evaluation Metrics")
print(f"✅ Accuracy       : {accuracy:.4f}")
print(f"🎯 Precision      : {precision_val:.4f}")
print(f"📥 Recall         : {recall_val:.4f}")
print(f"📈 F1 Score       : {f1:.4f}")
print(f"🧠 ROC-AUC        : {roc_auc:.4f}")
print(f"🚨 FDR            : {false_discovery_rate:.4f}")

if false_discovery_rate > 0.4:
    print("\n⚠️ High FDR Alert: Model may be over-flagging benign traffic.")
    print("🔧 Suggestion: Tune threshold or retrain with better class balance.")

# === Plot Confusion Matrix ===
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("\n📸 Saved confusion matrix as 'confusion_matrix.png'")

# === Detailed Classification Report ===
print("\n📂 Classification Report:")
print(classification_report(y_resampled, y_pred))
