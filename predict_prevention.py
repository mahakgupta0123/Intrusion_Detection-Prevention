import pandas as pd
import numpy as np
from stable_baselines3 import PPO

# === Config ===
TOP_FEATURES = ['sttl', 'ct_dst_sport_ltm', 'rate', 'proto', 'ct_srv_dst',
                'service', 'smean', 'ct_state_ttl', 'sbytes', 'ct_srv_src']
MODEL_PATH = 'models/prevention_dqn_model.zip'
X_TEST_PATH = 'data/processed/X_test_unsw.csv'

# === Load Model ===
model = PPO.load(MODEL_PATH)

# === Load Already-Scaled Test Data ===
X_test = pd.read_csv(X_TEST_PATH)

# === Ensure all top features exist ===
missing = [f for f in TOP_FEATURES if f not in X_test.columns]
if missing:
    raise ValueError(f"❌ Missing features in test data: {missing}")

X_test = X_test[TOP_FEATURES]

# === Prediction Function ===
def predict_single_row(input_row: pd.Series):
    state = np.array(input_row).astype(np.float32).reshape(1, -1)
    action, _ = model.predict(state, deterministic=True)
    return "BLOCK" if action[0] == 1 else "ALLOW"

# === Run on Some Flows ===
print("\n🔍 RL Prevention Decision on 5 flows:\n")
for i in range(5):
    row = X_test.iloc[i]
    decision = predict_single_row(row)
    print(f"Flow {i+1}: {decision}")
