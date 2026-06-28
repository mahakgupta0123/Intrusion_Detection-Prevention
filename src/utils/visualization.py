"""Visualization utilities"""

import matplotlib.pyplot as plt
import seaborn as sns


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
        plt.savefig(save_path, dpi=300)
    
    return plt


def plot_training_history(history, metrics=['loss', 'accuracy'], save_path=None):
    """Plot training history from Keras history object"""
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
        plt.savefig(save_path, dpi=300)
    
    return plt
