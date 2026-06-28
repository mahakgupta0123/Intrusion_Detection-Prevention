"""
Advanced Data Augmentation Techniques (SMOTE, ADASYN, etc.)

Combines GAN augmentation with SMOTE for optimal class balance.
"""

import pandas as pd
import numpy as np
import os
import logging
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class DataAugmentationPipeline:
    """Complete data augmentation pipeline"""
    
    def __init__(self, gan_multiplier=1.5, smote_ratio=0.8):
        self.gan_multiplier = gan_multiplier
        self.smote_ratio = smote_ratio
        
    def apply_gan_augmentation(self, X_train, y_train, generator):
        """Apply GAN augmentation"""
        from .gan_augmentor import GANAugmentor
        augmentor = GANAugmentor()
        augmentor.generator = generator
        return augmentor.augment_data(X_train, y_train, self.gan_multiplier)
    
    def apply_smote_augmentation(self, X_train, y_train):
        """Apply SMOTE augmentation"""
        return apply_smote_augmentation(X_train, y_train, sampling_strategy=self.smote_ratio)
    
    def apply_adasyn_augmentation(self, X_train, y_train):
        """Apply ADASYN augmentation"""
        return apply_adasyn_augmentation(X_train, y_train, sampling_strategy=self.smote_ratio)
    
    def combine_gan_and_smote(self, X_gan, y_gan):
        """Combine GAN with SMOTE"""
        return combine_gan_and_smote(X_gan, y_gan, apply_smote=True, smote_ratio=self.smote_ratio)


def apply_smote_augmentation(X_train, y_train, sampling_strategy='minority', k_neighbors=5, random_state=42):
    """
    Apply SMOTE for class balance
    
    Args:
        X_train: Training features
        y_train: Training labels
        sampling_strategy: 'minority' or float (0.0-1.0)
        k_neighbors: Number of nearest neighbors
        random_state: Random seed
    
    Returns:
        X_smote, y_smote: Augmented data
    """
    logger.info(f"🔄 Applying SMOTE augmentation...")
    logger.info(f"   Original: {pd.Series(y_train).value_counts().to_dict()}")
    
    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    
    logger.info(f"   After SMOTE: {pd.Series(y_smote).value_counts().to_dict()}")
    ratio = (pd.Series(y_smote) == 1).sum() / (pd.Series(y_smote) == 0).sum()
    logger.info(f"   Minority/majority ratio: {ratio:.4f}")
    
    return pd.DataFrame(X_smote, columns=X_train.columns), pd.Series(y_smote)


def apply_adasyn_augmentation(X_train, y_train, sampling_strategy='minority', n_neighbors=5, random_state=42):
    """
    Apply ADASYN (Adaptive Synthetic Sampling)
    
    Adaptive version that generates more samples for hard-to-learn cases.
    """
    logger.info(f"🔄 Applying ADASYN augmentation...")
    logger.info(f"   Original: {pd.Series(y_train).value_counts().to_dict()}")
    
    adasyn = ADASYN(sampling_strategy=sampling_strategy, n_neighbors=n_neighbors, random_state=random_state)
    X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
    
    logger.info(f"   After ADASYN: {pd.Series(y_adasyn).value_counts().to_dict()}")
    
    return pd.DataFrame(X_adasyn, columns=X_train.columns), pd.Series(y_adasyn)


def combine_gan_and_smote(X_gan_augmented, y_gan_augmented, apply_smote=True, smote_ratio=0.8):
    """
    Combine GAN-augmented data with SMOTE
    
    GAN captures attack patterns, SMOTE handles density.
    
    Args:
        X_gan_augmented: GAN-augmented features
        y_gan_augmented: GAN-augmented labels
        apply_smote: Whether to apply SMOTE
        smote_ratio: SMOTE sampling strategy (e.g., 0.8 = 80% of majority)
    
    Returns:
        X_final, y_final: Combined augmented data
    """
    logger.info(f"🔀 Combining GAN + SMOTE...")
    logger.info(f"   After GAN: {y_gan_augmented.value_counts().to_dict()}")
    
    if apply_smote:
        X_final, y_final = apply_smote_augmentation(
            X_gan_augmented, y_gan_augmented,
            sampling_strategy=smote_ratio
        )
    else:
        X_final = X_gan_augmented
        y_final = y_gan_augmented
    
    logger.info(f"   Final distribution: {pd.Series(y_final).value_counts().to_dict()}")
    
    return X_final, y_final


def compute_synthetic_quality_metrics(real_data, synthetic_data):
    """
    Compute quality metrics for synthetic data
    
    Returns:
        dict with KL-divergence, Wasserstein distance, etc.
    """
    metrics = {}
    
    # KL-Divergence per feature
    kl_divs = []
    for i in range(real_data.shape[1]):
        real_hist, bins = np.histogram(real_data[:, i], bins=20, density=True)
        synth_hist, _ = np.histogram(synthetic_data[:, i], bins=bins, density=True)
        
        real_hist = real_hist / (np.sum(real_hist) + 1e-10)
        synth_hist = synth_hist / (np.sum(synth_hist) + 1e-10)
        
        kl = np.sum(real_hist * np.log((real_hist + 1e-10) / (synth_hist + 1e-10)))
        kl_divs.append(kl)
    
    metrics['avg_kl_divergence'] = np.mean(kl_divs)
    metrics['max_kl_divergence'] = np.max(kl_divs)
    
    # Mean and std difference
    metrics['mean_abs_diff'] = np.abs(np.mean(real_data, axis=0) - np.mean(synthetic_data, axis=0)).mean()
    metrics['std_abs_diff'] = np.abs(np.std(real_data, axis=0) - np.std(synthetic_data, axis=0)).mean()
    
    # Wasserstein-like distance
    metrics['wasserstein_approx'] = np.mean([
        np.abs(np.percentile(real_data[:, i], 50) - np.percentile(synthetic_data[:, i], 50))
        for i in range(real_data.shape[1])
    ])
    
    return metrics


def validate_augmented_data(X_original, y_original, X_augmented, y_augmented, test_size=0.2):
    """
    Validate synthetic data quality
    
    Trains on original, tests on synthetic to check if synthetic data is realistic.
    """
    logger.info(f"🔍 Validating augmented data...")
    
    n_original = len(X_original)
    X_synthetic_only = X_augmented[n_original:] if len(X_augmented) > n_original else X_augmented
    y_synthetic_only = y_augmented[n_original:] if len(y_augmented) > n_original else y_augmented
    
    if len(X_synthetic_only) == 0:
        logger.warning("No synthetic samples to validate")
        return {}
    
    # Train on original, test on synthetic
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_original, y_original)
    
    y_pred = model.predict(X_synthetic_only)
    y_proba = model.predict_proba(X_synthetic_only)
    confidence = np.max(y_proba, axis=1).mean()
    
    logger.info(f"   Model confidence on synthetic: {confidence:.4f}")
    logger.info(f"   Synthetic samples as attacks: {(y_pred == 1).sum()}/{len(y_pred)}")
    
    if confidence < 0.5:
        logger.warning("Low confidence suggests synthetic data may be unrealistic")
    else:
        logger.info("✅ Synthetic data quality looks good")
    
    return {
        'model_confidence': confidence,
        'attack_rate_in_synthetic': (y_pred == 1).sum() / len(y_pred)
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Advanced augmentation module loaded")
    logger.info("Use: apply_smote_augmentation, apply_adasyn_augmentation, combine_gan_and_smote")
