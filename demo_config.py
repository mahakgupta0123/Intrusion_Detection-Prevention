# demo_config.py
import os

# --- Logging Configuration for Demo ---
LOG_DIR = "logs" # Logs will be saved in a 'logs' folder
DETECTION_LOG_FILE = os.path.join(LOG_DIR, "demo_detection_events.log")
PREVENTION_LOG_FILE = os.path.join(LOG_DIR, "demo_prevention_actions.log")
LOG_LEVEL = "INFO" # INFO for standard output, DEBUG for more verbose

# --- Model Paths ---
# IMPORTANT: Adjust 'MODEL_DIR' if your models are not in a 'models' sub-directory
# If your .pkl and .csv files are directly in your main project folder, set MODEL_DIR = "."
# MODEL_DIR = "trained_models" # Assuming your models are in a 'models' directory relative to script
# Example: If your main folder is 'my_ips_project' and models are in 'my_ips_project/models', then this is correct.
# If models are directly in 'my_ips_project', then set MODEL_DIR = "."

MODEL_DIR = 'trained_models'
XGB_MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_model_unsw.pkl')
PROCESSED_DATA_DIR = 'data/processed'
SCALER_PATH = os.path.join(PROCESSED_DATA_DIR, 'unsw_scaler.pkl')
LABEL_ENCODERS_PATH = os.path.join(PROCESSED_DATA_DIR, 'unsw_label_encoders.pkl')

# --- Feature Extraction Configuration ---
FLOW_IDLE_TIMEOUT = 120  # seconds (how long before a flow is considered idle)
FLOW_ACTIVE_TIMEOUT = 300 # seconds (max duration for a flow)

# --- Prediction Configuration ---
PREDICTION_PACKET_INTERVAL = 5 # Make a prediction every 5 packets in an active flow
PREVENTION_THRESHOLD = 0.8 # Probability threshold (0.0-1.0) to trigger an attack detection

# --- IPS Actions Configuration ---
ENABLE_PREVENTION_ACTIONS = False # IMPORTANT: Set to False for this demo with PCAP. This simulates blocks.
BLOCK_DURATION_SECONDS = 300 # How long to block an IP (if ENABLE_PREVENTION_ACTIONS is True)

# --- Network Interface (Not used when reading from PCAP, but required by code) ---
NETWORK_INTERFACE = "eth0" # Placeholder. The PCAP path will override this for the demo.