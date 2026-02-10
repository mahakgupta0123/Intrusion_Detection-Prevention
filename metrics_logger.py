# metrics_logger.py

import json
import os

METRICS_FILE = "metrics.json"
metrics = {
    "total_flows": 0,
    "attacks_detected": 0,
    "blocks_performed": 0
}

# Load existing metrics if file exists (optional but recommended)
if os.path.exists(METRICS_FILE):
    try:
        with open(METRICS_FILE, "r") as f:
            saved = json.load(f)
            metrics.update(saved)
    except Exception:
        pass  # fallback to default values


def save_metrics():
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4)


def update_metric(metric_name):
    if metric_name in metrics:
        metrics[metric_name] += 1
    else:
        metrics[metric_name] = 1

    save_metrics()  # 🔥 THIS ENSURES REAL-TIME SAVE
