import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from datetime import datetime

# =========================
# Load Top SHAP Features
# =========================
def get_top_features(shap_csv_path, top_n=5):
    shap_df = pd.read_csv(shap_csv_path)
    return shap_df['feature'].head(top_n).tolist()

# =========================
# Apply Prevention Rules
# =========================
def apply_prevention_rules(df, top_features):
    print("⚙️  Applying prevention rules...")
    rules = {
        top_features[0]: lambda x: x > 0.7,
        top_features[1]: lambda x: x < -0.5,
        top_features[2]: lambda x: x > 1.2,
        top_features[3]: lambda x: x < 0.3,
        top_features[4]: lambda x: x > 0.8,
    }

    df['prevent_flag'] = 0
    for feat, rule in rules.items():
        df['prevent_flag'] |= df[feat].apply(rule).astype(int)
    return df

# =========================
# Train XGBoost Model
# =========================
def train_prevention_model(X, y, top_features, save_path):
    print("🧠 Training XGBoost prevention model...")
    X_subset = X[top_features]
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_subset, y)
    joblib.dump(model, save_path)
    print(f"✅ Model saved to {save_path}")
    return model

# =========================
# Agentic AI Class
# =========================
class AgenticAIPrevention:
    def __init__(self, rules, log_file='results/agent_decisions.log'):
        self.rules = rules
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(self.log_file, 'w') as f:
            f.write("Time,Feature,Value,Decision\n")

    def decide(self, sample):
        decisions = []
        for feat, cond in self.rules.items():
            val = sample[feat]
            if cond(val):
                decision = "BLOCKED"
                decisions.append((feat, val, decision))
                self._log_decision(feat, val, decision)
        return decisions

    def _log_decision(self, feat, val, decision):
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()},{feat},{val},{decision}\n")

# =========================
# Main Pipeline
# =========================
if __name__ == "__main__":
    print("🚀 Starting Prevention Layer Pipeline...\n")

    # Load data
    X = pd.read_csv("data/processed/X_train_unsw.csv")
    y = pd.read_csv("data/processed/y_train_unsw.csv").squeeze()

    # Get top SHAP features
    shap_path = "results/xai_explanations/shap_values_unsw.csv"
    top_features = get_top_features(shap_path, top_n=5)
    print(f"📌 Top features: {top_features}")

    # Apply rules
    flagged_df = apply_prevention_rules(X.copy(), top_features)
    os.makedirs("results/prevention", exist_ok=True)
    flagged_df.to_csv("results/prevention/flagged_unsw.csv", index=False)
    print("✅ Flagged data saved.")

    # Train model on top features
    model_save_path = "trained_models/prevention_model_unsw.pkl"
    model = train_prevention_model(X, y, top_features, model_save_path)

    # Run Agentic AI
    print("\n🤖 Running Agentic AI on first 5 samples...")
    agent = AgenticAIPrevention({
        feat: lambda x: x > X[feat].quantile(0.85) for feat in top_features
    })

    for i in range(5):
        decisions = agent.decide(X.iloc[i])
        if decisions:
            print(f"🚨 Sample {i}: Agentic prevention triggered -> {decisions}")
        else:
            print(f"✅ Sample {i}: No prevention needed.")
