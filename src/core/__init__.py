"""Core IDS/IPS functionality"""

from .prediction_engine import PredictionEngine
from .real_time_monitoring import RealTimeIPS

__all__ = [
    'PredictionEngine',
    'RealTimeIPS'
]
