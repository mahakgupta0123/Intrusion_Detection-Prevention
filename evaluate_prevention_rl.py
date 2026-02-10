import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from stable_baselines3 import PPO
import time
from sklearn.preprocessing import StandardScaler
import joblib

# === Config ===
MODEL_PATH = 'models/prevention_ppo_model.zip'
X_TEST_PATH = 'data/processed/X_test_unsw.csv'
Y_TEST_PATH = 'data/processed/y_test_unsw.csv'
SCALER_PATH = 'models/prevention_scaler.pkl'
ENCODER_PATHS = {
    'proto': 'models/proto_encoder.pkl',
    'service': 'models/service_encoder.pkl'
}
TOP_FEATURES = ['sttl', 'ct_dst_sport_ltm', 'rate', 'proto', 'ct_srv_dst',
                'service', 'smean', 'ct_state_ttl', 'sbytes', 'ct_srv_src']

# === Load Model ===
model = PPO.load(MODEL_PATH)

# === Load Data ===
X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH)
X_test = X_test[TOP_FEATURES].copy()

# Apply encoders
for col in ['proto', 'service']:
    le = joblib.load(ENCODER_PATHS[col])
    X_test[col] = le.transform(X_test[col].astype(str))

# Scale features
scaler = joblib.load(SCALER_PATH)
X_test = scaler.transform(X_test)

# === Predict and Time ===
predictions = []
latencies = []

for i in range(len(X_test)):
    obs = X_test[i].astype(np.float32)
    start_time = time.time()
    action, _ = model.predict(obs, deterministic=True)
    end_time = time.time()
    predictions.append(action)
    latencies.append(end_time - start_time)

avg_latency_ms = np.mean(latencies) * 1000

# === Results ===
print("=== Evaluation Results ===")
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions, target_names=["Benign", "Attack"]))
print(f"\n🕒 Average Decision Latency: {avg_latency_ms:.4f} ms")
