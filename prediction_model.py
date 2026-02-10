# prediction_model.py

import joblib
import numpy as np
import subprocess

# Load your trained model (place model.pkl in the same directory)
model = joblib.load("model.pkl")

def predict_packet(features):
    """
    Predict if a packet is malicious using the trained model.
    Returns: (prediction_label, confidence_score)
    """
    X = np.array(features).reshape(1, -1)  # reshape for a single sample
    probas = model.predict_proba(X)[0]
    classes = model.classes_

    prediction_index = np.argmax(probas)
    prediction = classes[prediction_index]
    confidence = probas[prediction_index]

    return prediction, confidence

def block_source_ip(ip_address):
    """
    Block the IP using iptables (Linux firewall).
    You must run the Docker container with NET_ADMIN capability.
    """
    try:
        subprocess.run(["iptables", "-A", "INPUT", "-s", ip_address, "-j", "DROP"], check=True)
        print(f"[✅] Blocked IP: {ip_address}")
    except subprocess.CalledProcessError as e:
        print(f"[❌] Failed to block IP {ip_address}: {e}")
