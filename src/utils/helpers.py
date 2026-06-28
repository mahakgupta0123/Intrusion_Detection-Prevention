"""Utility helper functions"""

import os
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

logger = logging.getLogger(__name__)


def ensure_directory(path):
    """Create directory if not exists"""
    os.makedirs(path, exist_ok=True)
    return path


def load_model(path):
    """Load pickled model"""
    model = joblib.load(path)
    logger.info(f"✅ Model loaded from {path}")
    return model


def save_model(model, path):
    """Save model to pickle"""
    ensure_directory(os.path.dirname(path))
    joblib.dump(model, path)
    logger.info(f"✅ Model saved to {path}")


def plot_confusion_matrix(cm, labels=['Normal', 'Attack'], title='Confusion Matrix', 
                         save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    if save_path:
        ensure_directory(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300)
        logger.info(f"✅ Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_training_history(history, metrics=['loss', 'accuracy'], save_path=None):
    """Plot training history"""
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx] if len(metrics) > 1 else axes
        
        ax.plot(history.history[metric], label='Train')
        if f'val_{metric}' in history.history:
            ax.plot(history.history[f'val_{metric}'], label='Val')
        
        ax.set_title(f'{metric.capitalize()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        ensure_directory(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300)
        logger.info(f"✅ Training history saved to {save_path}")
    
    plt.show()
