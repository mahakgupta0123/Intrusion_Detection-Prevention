"""Metrics logging utility"""

import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Log and track metrics across experiments"""
    
    def __init__(self, log_dir='results/logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics = {}
    
    def log_metrics(self, run_name, metrics, model_type='unknown'):
        """Log metrics for a run"""
        self.metrics[run_name] = {
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'metrics': metrics
        }
        logger.info(f"✅ Metrics logged for {run_name}")
    
    def save_json(self, filename='metrics.json'):
        """Save metrics to JSON"""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"✅ Metrics saved to {filepath}")
    
    def load_json(self, filename='metrics.json'):
        """Load metrics from JSON"""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'r') as f:
            self.metrics = json.load(f)
        logger.info(f"✅ Metrics loaded from {filepath}")
    
    def print_summary(self):
        """Print metrics summary"""
        logger.info("\n📊 Metrics Summary:")
        for run_name, data in self.metrics.items():
            logger.info(f"\n  {run_name}:")
            logger.info(f"    Model: {data['model_type']}")
            logger.info(f"    Time: {data['timestamp']}")
            for metric, value in data['metrics'].items():
                if isinstance(value, (int, float)):
                    logger.info(f"    {metric}: {value:.4f}")
