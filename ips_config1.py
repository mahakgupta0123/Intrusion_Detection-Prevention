# ips_config.py

import os
import yaml
import logging
from scapy.all import get_if_list

# Load YAML configuration
CONFIG_PATH = 'config.yaml'
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Missing {CONFIG_PATH}")

with open(CONFIG_PATH) as file:
    config = yaml.safe_load(file)

# Logging config
LOG_LEVEL = config.get('logging', {}).get('level', 'INFO')

# Detect available network interfaces
def get_available_interfaces():
    interfaces = get_if_list()
    return [i for i in interfaces if i != 'lo']

# Detect default interface
def detect_interface():
    configured = config.get('network', {}).get('interface', 'auto')
    if configured != 'auto':
        return configured
    interfaces = get_available_interfaces()
    return interfaces[0] if interfaces else 'lo'

NETWORK_INTERFACE = detect_interface()

# Prevention settings
ENABLE_PREVENTION_ACTIONS = config.get('prevention', {}).get('enable', True)
PREVENTION_THRESHOLD = config.get('prevention', {}).get('threshold', 0.85)
BLOCK_DURATION_SECONDS = config.get('prevention', {}).get('block_duration', 300)

# Other constants
MODEL_DIR = 'trained_models'
XGB_MODEL_PATH = os.path.join(MODEL_DIR, 'xgb_model_unsw.pkl')
PROCESSED_DATA_DIR = 'data/processed'
SCALER_PATH = os.path.join(PROCESSED_DATA_DIR, 'unsw_scaler.pkl')
LABEL_ENCODERS_PATH = os.path.join(PROCESSED_DATA_DIR, 'unsw_label_encoders.pkl')
LOG_DIR = 'ips_logs'
os.makedirs(LOG_DIR, exist_ok=True)
DETECTION_LOG_FILE = os.path.join(LOG_DIR, 'detection_events.log')
PREVENTION_LOG_FILE = os.path.join(LOG_DIR, 'prevention_actions.log')

print(f"🌐 Interface in use: {NETWORK_INTERFACE}")
print(f"🛡️  Prevention Enabled: {ENABLE_PREVENTION_ACTIONS}")
