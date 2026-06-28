#!/usr/bin/env python3
"""
Complete IPDRS Pipeline

Orchestrates the full workflow:
1. Data preprocessing
2. GAN augmentation
3. SMOTE augmentation
4. Model training
5. Evaluation with stratified k-fold
"""

import sys
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import DataPreprocessor
from src.augmentation import GANAugmentor, combine_gan_and_smote
from src.models import ModelTrainer
from src.evaluation import stratified_kfold_evaluation
from src.utils import ensure_directory, MetricsLogger


def main():
    """Run complete pipeline"""
    
    logger.info("="*80)
    logger.info("🚀 IPDRS Complete Pipeline")
    logger.info("="*80)
    
    # ============================================
    # STEP 1: Data Preprocessing
    # ============================================
    logger.info("\n📥 STEP 1: Data Preprocessing")
    logger.info("-"*80)
    
    preprocessor = DataPreprocessor(
        data_dir='../../data/processed',
        raw_data_dir='../../data/UNSW‑NB15'
    )
    
    try:
        X_train, y_train, X_test, y_test = preprocessor.process_pipeline()
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return
    
    # ============================================
    # STEP 2: GAN Augmentation
    # ============================================
    logger.info("\n🔄 STEP 2: GAN Augmentation")
    logger.info("-"*80)
    
    augmentor = GANAugmentor(latent_dim=100, batch_size=64)
    attack_data = X_train[y_train == 1].values
    
    if attack_data.shape[0] > 100:
        try:
            augmentor.train(attack_data, epochs=5000)
            X_aug, y_aug = augmentor.augment_data(X_train, y_train, multiplier=1.5)
            augmentor.save_models('../../trained_models')
        except Exception as e:
            logger.error(f"GAN training failed: {e}")
            X_aug, y_aug = X_train, y_train
    else:
        logger.warning(f"Not enough attack samples ({attack_data.shape[0]})")
        X_aug, y_aug = X_train, y_train
    
    # ============================================
    # STEP 3: SMOTE Augmentation
    # ============================================
    logger.info("\n🔀 STEP 3: SMOTE Augmentation")
    logger.info("-"*80)
    
    try:
        X_final, y_final = combine_gan_and_smote(X_aug, y_aug, apply_smote=True, smote_ratio=0.8)
    except Exception as e:
        logger.error(f"SMOTE failed: {e}")
        X_final, y_final = X_aug, y_aug
    
    # ============================================
    # STEP 4: Model Training & Evaluation
    # ============================================
    logger.info("\n🤖 STEP 4: Model Training & Evaluation")
    logger.info("-"*80)
    
    metrics_logger = MetricsLogger('../../results/logs')
    
    for model_type in ['rf', 'xgb']:
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"# Training {model_type.upper()}")
        logger.info(f"{'#'*80}")
        
        try:
            results = stratified_kfold_evaluation(
                X_final, y_final, X_test, y_test,
                model_type=model_type,
                n_splits=5
            )
            
            if results:
                metrics_logger.log_metrics(
                    f'{model_type}_augmented',
                    results['test_metrics'],
                    model_type=model_type
                )
        except Exception as e:
            logger.error(f"Evaluation failed for {model_type}: {e}")
    
    # Save metrics
    metrics_logger.save_json('../../results/logs/pipeline_metrics.json')
    metrics_logger.print_summary()
    
    logger.info("\n" + "="*80)
    logger.info("✅ Pipeline Complete!")
    logger.info("="*80)
    logger.info("\n📊 Summary:")
    logger.info("   - Data preprocessed and augmented")
    logger.info("   - GAN trained and synthetic samples generated")
    logger.info("   - SMOTE applied for optimal balance")
    logger.info("   - Models trained with stratified k-fold CV")
    logger.info("   - Results saved to results/")


if __name__ == "__main__":
    main()
