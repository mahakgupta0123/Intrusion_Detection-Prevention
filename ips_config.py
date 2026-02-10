# ips_config.py

import os

# --- General Paths ---

MODEL_DIR = 'trained_models'

PROCESSED_DATA_DIR = 'data/processed'

LOG_DIR = 'ips_logs' # Directory for IPS operational logs

PREVENTION_LOG_FILE = os.path.join(LOG_DIR, 'prevention_actions.log') # Log for actions taken

DETECTION_LOG_FILE = os.path.join(LOG_DIR, 'detection_events.log') # Log for detections (even if no action)

# Ensure log directory exists

os.makedirs(LOG_DIR, exist_ok=True)

# --- Model Files ---

XGB_MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_model_unsw.pkl')

SCALER_PATH = os.path.join(PROCESSED_DATA_DIR, 'unsw_scaler.pkl')

LABEL_ENCODERS_PATH = os.path.join(PROCESSED_DATA_DIR, 'unsw_label_encoders.pkl')

# --- Network Interface Configuration ---

# IMPORTANT: Replace with your actual network interface name

# Use ifconfig or ip addr on Linux to find this.

# This should be the interface through which you want to monitor/prevent traffic.

NETWORK_INTERFACE = "eth0"

# e.g., "eth0", "wlan0", "enp0s3"

# --- IPS Thresholds and Behavior ---

# Probability threshold for an "attack" prediction from the XGBoost model

# Higher values = fewer false positives but potentially more missed attacks

PREVENTION_THRESHOLD = 0.85 # Start high for prevention. Adjust carefully!

# Time window (in seconds) to consider a "flow" for feature calculation

# UNSW-NB15 features are aggregated over flows.

FLOW_IDLE_TIMEOUT = 10 # If no packets for a flow for this duration, consider flow ended.

FLOW_ACTIVE_TIMEOUT = 120 # Maximum duration to keep a flow active for feature aggregation.

# How frequently (in packets per flow) to run prediction on an active flow

PREDICTION_PACKET_INTERVAL = 5

# --- Prevention Actions ---

# Whether to actually execute iptables commands or just log them (for testing)

# SET TO FALSE FOR TESTING! ONLY TRUE IN CONTROLLED ENVIRONMENT!

ENABLE_PREVENTION_ACTIONS = False

# Duration for temporary firewall rules (in seconds)

# Set to 0 for permanent block (use with extreme caution!)

BLOCK_DURATION_SECONDS = 300 # 5 minutes

# --- Logging Level ---

# Set to 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

LOG_LEVEL = 'INFO'
