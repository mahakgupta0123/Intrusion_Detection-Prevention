"""
IPDRS (Intrusion Prevention and Detection Research System)

A comprehensive system for network intrusion detection and prevention using:
- GAN-based data augmentation for handling class imbalance
- Multiple ML models (Random Forest, XGBoost, LSTM, CNN)
- Advanced evaluation with stratified k-fold cross-validation
- Real-time IPS capabilities
- Explainability with SHAP and LIME
- Prevention models and RL-based mitigation
- REST API deployment
"""

__version__ = "1.0.0"
__author__ = "IPDRS Research Team"

# Core modules
from .preprocessing import DataPreprocessor
from .augmentation import GANAugmentor, DataAugmentationPipeline
from .models import ModelTrainer
from .evaluation import ModelEvaluator
from .core import PredictionEngine, RealTimeIPS

# Extended modules
from .xai import SHAPExplainer, LIMEExplainer, generate_shap_csv
from .features import FeatureExtractor
from .prevention import PreventionModel, PreventionEnv
from .deployment import create_app

__all__ = [
    # Core
    'DataPreprocessor',
    'GANAugmentor',
    'DataAugmentationPipeline',
    'ModelTrainer',
    'ModelEvaluator',
    'PredictionEngine',
    'RealTimeIPS',
    # XAI
    'SHAPExplainer',
    'LIMEExplainer',
    'generate_shap_csv',
    # Features
    'FeatureExtractor',
    # Prevention
    'PreventionModel',
    'PreventionEnv',
    # Deployment
    'create_app'
]
