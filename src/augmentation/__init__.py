"""Data augmentation module with GAN and SMOTE techniques"""

from .gan_augmentor import GANAugmentor, build_generator, build_discriminator, train_gan
from .advanced_augmentation import (
    DataAugmentationPipeline,
    apply_smote_augmentation,
    apply_adasyn_augmentation,
    combine_gan_and_smote,
    compute_synthetic_quality_metrics,
    validate_augmented_data
)

__all__ = [
    'GANAugmentor',
    'build_generator',
    'build_discriminator',
    'train_gan',
    'DataAugmentationPipeline',
    'apply_smote_augmentation',
    'apply_adasyn_augmentation',
    'combine_gan_and_smote',
    'compute_synthetic_quality_metrics',
    'validate_augmented_data'
]
