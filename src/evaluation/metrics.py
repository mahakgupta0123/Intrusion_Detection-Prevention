"""Metrics computation utilities"""

import logging

logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred, y_proba=None):
    """Compute evaluation metrics"""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, 
        roc_auc_score, confusion_matrix
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    
    metrics['cm'] = confusion_matrix(y_true, y_pred)
    
    return metrics


def print_metrics(metrics):
    """Pretty print metrics"""
    logger.info("\n📊 Metrics:")
    for key, value in metrics.items():
        if key != 'cm':
            logger.info(f"  {key}: {value:.4f}")
