import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env_prevention import PreventionEnv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# === Config ===
TOP_FEATURES = ['sttl', 'ct_dst_sport_ltm', 'rate', 'proto', 'ct_srv_dst',
                'service', 'smean', 'ct_state_ttl', 'sbytes', 'ct_srv_src']
DATA_DIR = 'data/processed'
MODEL_SAVE_PATH = 'models/prevention_ppo_model.zip'
SCALER_PATH = 'models/prevention_scaler.pkl'
ENCODER_PATHS = {
    'proto': 'models/proto_encoder.pkl',
    'service': 'models/service_encoder.pkl'
}

# === Load Processed Data ===
X_train = pd.read_csv(f'{DATA_DIR}/X_train_unsw.csv')
y_train = pd.read_csv(f'{DATA_DIR}/y_train_unsw.csv')

# Combine for balancing
df = X_train.copy()
df['label'] = y_train

# Separate classes
benign = df[df['label'] == 0]
attack = df[df['label'] == 1]

# Downsample attack to match benign
attack_balanced = attack.sample(n=len(benign), random_state=42)
balanced_df = pd.concat([benign, attack_balanced])

# Shuffle and extract
balanced_df = balanced_df.sample(frac=1, random_state=42)
X_selected_raw = balanced_df[TOP_FEATURES].copy()

# Encode categorical features
for col in ['proto', 'service']:
    le = LabelEncoder()
    X_selected_raw[col] = le.fit_transform(X_selected_raw[col].astype(str))
    joblib.dump(le, ENCODER_PATHS[col])

# Scale features
scaler = StandardScaler()
X_selected = scaler.fit_transform(X_selected_raw)
joblib.dump(scaler, SCALER_PATH)

y_selected = balanced_df['label']

# === Optional Split for Testing ===
X_train_small, _, y_train_small, _ = train_test_split(X_selected, y_selected, test_size=0.1, random_state=42)

# === RL Environment ===
env = PreventionEnv(X_selected, y_selected)

# === Check Environment ===
check_env(env, warn=True)

# === Train PPO Model ===
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    batch_size=128,
    n_steps=1024,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    tensorboard_log="./tensorboard_logs/"
)

model.learn(total_timesteps=200_000)
model.save(MODEL_SAVE_PATH)
print(f"\n✅ Model saved at: {MODEL_SAVE_PATH}")
