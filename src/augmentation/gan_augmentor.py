"""
GAN-based Data Augmentation Module

Improves class imbalance by generating synthetic attack samples using a GAN.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import pandas as pd
import os
import joblib
import logging

logger = logging.getLogger(__name__)


class GANAugmentor:
    """GAN-based data augmentation for attack samples"""
    
    def __init__(self, latent_dim=100, batch_size=64, learning_rate=0.0002):
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.generator = None
        self.discriminator = None
        self.gan = None
        
    def build_generator(self, num_features):
        """Build generator network"""
        model = models.Sequential([
            layers.Dense(256, activation='relu', input_dim=self.latent_dim),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(num_features)
        ], name='generator')
        return model

    def build_discriminator(self, num_features):
        """Build discriminator network"""
        model = models.Sequential([
            layers.Dense(512, activation='relu', input_dim=num_features),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ], name='discriminator')
        return model

    def build_gan(self, generator, discriminator):
        """Build complete GAN model"""
        discriminator.trainable = False
        return models.Sequential([generator, discriminator], name='gan')

    def train(self, attack_data, epochs=5000):
        """
        Train GAN on attack samples
        
        Args:
            attack_data: Array of attack samples
            epochs: Number of training epochs
        """
        num_features = attack_data.shape[1]
        
        self.generator = self.build_generator(num_features)
        self.discriminator = self.build_discriminator(num_features)
        self.gan = self.build_gan(self.generator, self.discriminator)

        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(self.learning_rate, 0.5),
            metrics=['accuracy']
        )
        self.gan.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(self.learning_rate, 0.5))

        logger.info(f"Starting GAN training for {epochs} epochs...")
        d_losses = []
        g_losses = []

        half_batch = self.batch_size // 2

        for epoch in range(epochs):
            # Train discriminator
            idx = np.random.randint(0, attack_data.shape[0], half_batch)
            real_samples = attack_data[idx]
            noise = np.random.normal(0, 1, (half_batch, self.latent_dim))
            fake_samples = self.generator.predict(noise, verbose=0)

            real_labels = np.random.uniform(0.8, 1.0, (half_batch, 1))
            fake_labels = np.random.uniform(0.0, 0.2, (half_batch, 1))

            d_loss_real = self.discriminator.train_on_batch(real_samples, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_samples, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator
            noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
            gen_labels = np.random.uniform(0.8, 1.0, (self.batch_size, 1))
            g_loss = self.gan.train_on_batch(noise, gen_labels)

            d_losses.append(d_loss[0])
            g_losses.append(g_loss)

            if epoch % (epochs // 10) == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch:4d}/{epochs} | D Loss: {d_loss[0]:.4f} (Acc: {100*d_loss[1]:5.2f}%) | G Loss: {g_loss:.4f}")

        logger.info("✅ GAN training complete.")
        self._print_diagnostics(d_losses, g_losses)

    def _print_diagnostics(self, d_losses, g_losses):
        """Print training diagnostics"""
        logger.info(f"📊 Training Diagnostics:")
        logger.info(f"   Avg D Loss (last 100): {np.mean(d_losses[-100:]):.4f}")
        logger.info(f"   Avg G Loss (last 100): {np.mean(g_losses[-100:]):.4f}")
        logger.info(f"   Final D Loss: {d_losses[-1]:.4f}")
        logger.info(f"   Final G Loss: {g_losses[-1]:.4f}")

        if d_losses[-1] > 1.0:
            logger.warning("Discriminator loss too high (>1.0)")
        if g_losses[-1] > 5.0:
            logger.warning("Generator loss high (>5.0)")
        if np.mean(g_losses[-100:]) > np.mean(g_losses[-200:-100]):
            logger.warning("Possible mode collapse - G loss increasing")
        else:
            logger.info("✅ Generator showing improvement")

    def generate_synthetic_samples(self, num_samples):
        """Generate synthetic attack samples"""
        if self.generator is None:
            raise ValueError("GAN not trained yet. Call train() first.")
        
        noise = np.random.normal(0, 1, (num_samples, self.latent_dim))
        return self.generator.predict(noise, verbose=0)

    def augment_data(self, X_train, y_train, multiplier=1.5, use_adaptive_clipping=True):
        """
        Generate and augment data with synthetic attack samples
        
        Args:
            X_train: Training features
            y_train: Training labels
            multiplier: Multiplier for synthetic samples (e.g., 1.5x)
            use_adaptive_clipping: Use percentile-based clipping
        """
        attack_data = X_train[y_train == 1].values
        if attack_data.shape[0] == 0:
            logger.warning("No attack samples found. Skipping augmentation.")
            return X_train, y_train

        num_synthetic = max(1, int(attack_data.shape[0] * multiplier))
        synthetic_attacks = self.generate_synthetic_samples(num_synthetic)

        # Adaptive clipping
        if use_adaptive_clipping:
            data_min = np.percentile(attack_data, 2.5)
            data_max = np.percentile(attack_data, 97.5)
            data_range = data_max - data_min

            extended_min = data_min - 0.1 * data_range
            extended_max = data_max + 0.1 * data_range

            synthetic_attacks = np.clip(synthetic_attacks, extended_min, extended_max)
            logger.info(f"Applied adaptive clipping: [{extended_min:.3f}, {extended_max:.3f}]")

        # Validate distribution
        kl_div = self._compute_kl_divergence(attack_data, synthetic_attacks)
        logger.info(f"KL Divergence (quality metric): {kl_div:.4f}")
        if kl_div > 0.5:
            logger.warning("High KL-divergence: synthetic data may not match distribution")
        else:
            logger.info("✅ Synthetic data distribution quality is good")

        # Create augmented dataset
        synthetic_df = pd.DataFrame(synthetic_attacks, columns=X_train.columns)
        synthetic_labels = pd.Series(1, index=synthetic_df.index)

        X_aug = pd.concat([X_train, synthetic_df], ignore_index=True)
        y_aug = pd.concat([y_train, synthetic_labels], ignore_index=True)

        logger.info(f"Generated {num_synthetic} synthetic samples ({multiplier}x multiplier)")
        logger.info(f"Augmented dataset: {X_aug.shape[0]} samples (original: {X_train.shape[0]})")
        
        return X_aug, y_aug

    @staticmethod
    def _compute_kl_divergence(real_data, synthetic_data, n_bins=20):
        """Compute KL-divergence between distributions"""
        kl_divs = []

        for i in range(real_data.shape[1]):
            real_hist, bin_edges = np.histogram(real_data[:, i], bins=n_bins, density=True)
            synth_hist, _ = np.histogram(synthetic_data[:, i], bins=bin_edges, density=True)

            real_hist = real_hist / (np.sum(real_hist) + 1e-10)
            synth_hist = synth_hist / (np.sum(synth_hist) + 1e-10)

            real_hist = np.maximum(real_hist, 1e-10)
            synth_hist = np.maximum(synth_hist, 1e-10)

            kl = np.sum(real_hist * np.log(real_hist / synth_hist))
            kl_divs.append(kl)

        return np.mean(kl_divs)

    def save_models(self, model_dir='trained_models'):
        """Save trained generator and discriminator"""
        os.makedirs(model_dir, exist_ok=True)
        if self.generator:
            self.generator.save(os.path.join(model_dir, 'generator_unsw.h5'))
        if self.discriminator:
            self.discriminator.save(os.path.join(model_dir, 'discriminator_unsw.h5'))
        logger.info(f"Models saved to {model_dir}")

    def load_models(self, model_dir='trained_models'):
        """Load pre-trained generator and discriminator"""
        self.generator = tf.keras.models.load_model(os.path.join(model_dir, 'generator_unsw.h5'))
        self.discriminator = tf.keras.models.load_model(os.path.join(model_dir, 'discriminator_unsw.h5'))
        logger.info(f"Models loaded from {model_dir}")


# Legacy functions for backward compatibility
def build_generator(latent_dim, num_features):
    """Build generator (legacy)"""
    augmentor = GANAugmentor(latent_dim=latent_dim)
    return augmentor.build_generator(num_features)


def build_discriminator(num_features):
    """Build discriminator (legacy)"""
    augmentor = GANAugmentor()
    return augmentor.build_discriminator(num_features)


def train_gan(generator, discriminator, gan, data, epochs=5000, batch_size=64, latent_dim=100):
    """Train GAN (legacy)"""
    augmentor = GANAugmentor(latent_dim=latent_dim, batch_size=batch_size)
    augmentor.generator = generator
    augmentor.discriminator = discriminator
    augmentor.gan = gan
    augmentor.train(data, epochs=epochs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    from ..preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    X_train, y_train, _, _ = preprocessor.process_pipeline()
    
    augmentor = GANAugmentor()
    attack_data = X_train[y_train == 1].values
    
    augmentor.train(attack_data, epochs=5000)
    X_aug, y_aug = augmentor.augment_data(X_train, y_train, multiplier=1.5)
    
    augmentor.save_models()
