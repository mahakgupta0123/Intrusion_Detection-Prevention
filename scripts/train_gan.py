#!/usr/bin/env python3
"""
GAN Training and Augmentation Script

Trains GAN and generates augmented data.
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
from src.augmentation import GANAugmentor


if __name__ == "__main__":
    logger.info("🔄 Loading preprocessed data...")
    
    preprocessor = DataPreprocessor(
        data_dir='data/processed',
        raw_data_dir='data/UNSW‑NB15'
    )
    
    try:
        X_train = __import__('pandas').read_csv('data/processed/X_train_unsw.csv')
        y_train = __import__('pandas').read_csv('data/processed/y_train_unsw.csv').squeeze()
    except FileNotFoundError:
        logger.error("Preprocessed data not found. Run preprocess_data.py first.")
        sys.exit(1)
    
    logger.info("🎯 Training GAN for augmentation...")
    
    augmentor = GANAugmentor(latent_dim=100, batch_size=64)
    attack_data = X_train[y_train == 1].values
    
    if attack_data.shape[0] > 100:
        try:
            augmentor.train(attack_data, epochs=5000)
            X_aug, y_aug = augmentor.augment_data(X_train, y_train, multiplier=1.5)
            
            # Save augmented data
            X_aug.to_csv('data/processed/X_train_unsw_augmented.csv', index=False)
            y_aug.to_csv('data/processed/y_train_unsw_augmented.csv', index=False)
            
            augmentor.save_models('trained_models')
            logger.info("✅ GAN training and augmentation complete!")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.error(f"Not enough attack samples: {attack_data.shape[0]}")
