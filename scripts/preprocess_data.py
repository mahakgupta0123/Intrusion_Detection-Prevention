#!/usr/bin/env python3
"""
Preprocessing-only Script

Runs only data preprocessing step.
"""

import sys
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import DataPreprocessor


if __name__ == "__main__":
    logger.info("🔄 Running Data Preprocessing...")
    
    preprocessor = DataPreprocessor(
        data_dir='data/processed',
        raw_data_dir='data/UNSW‑NB15'
    )
    
    try:
        X_train, y_train, X_test, y_test = preprocessor.process_pipeline()
        logger.info("✅ Preprocessing complete!")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
