"""
Prevention Module

This module provides prevention capabilities including:
- Prevention model training
- RL-based prevention environment
- Prevention strategies

Classes:
    PreventionModel - Train prevention models
    PreventionEnv - Gym environment for RL
    PreventionRL - RL-based prevention

Functions:
    train_prevention_model() - Train XGBoost prevention model
    apply_prevention_rules() - Apply prevention rules
    get_top_features() - Get important features

Usage:
    from src.prevention import PreventionModel
    
    model = PreventionModel()
    model.train(X_train, y_train)
    model.save('prevention_model.pkl')
"""

from .prevention_model import (
    PreventionModel,
    train_prevention_model,
    apply_prevention_rules,
    get_top_features
)

from .env_prevention import PreventionEnv

try:
    from .prevention_rl import PreventionRL
    HAS_RL = True
except ImportError:
    HAS_RL = False
    PreventionRL = None

__all__ = [
    'PreventionModel',
    'PreventionEnv',
    'train_prevention_model',
    'apply_prevention_rules',
    'get_top_features'
]

if HAS_RL:
    __all__.append('PreventionRL')
