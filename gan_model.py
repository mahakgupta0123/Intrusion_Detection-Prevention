import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import pandas as pd
import os
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

def build_generator(latent_dim, num_features):
    return models.Sequential([
        layers.Dense(256, activation='relu', input_dim=latent_dim),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(num_features)  
    ], name='generator')

def build_discriminator(num_features):
    return models.Sequential([
        layers.Dense(512, activation='relu', input_dim=num_features),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ], name='discriminator')

def build_gan(generator, discriminator):
    discriminator.trainable = False
    return models.Sequential([generator, discriminator], name='gan')

def train_gan(generator, discriminator, gan, data, epochs=5000, batch_size=64, latent_dim=100):
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])
    gan.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(0.0002, 0.5))

    half_batch = batch_size // 2
    print(f"Starting GAN training for {epochs} epochs...")

    for epoch in range(epochs):
        idx = np.random.randint(0, data.shape[0], half_batch)
        real_samples = data[idx]
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_samples = generator.predict(noise, verbose=0)

        d_loss_real = discriminator.train_on_batch(real_samples, np.ones((half_batch, 1)) * 0.9)
        d_loss_fake = discriminator.train_on_batch(fake_samples, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        if epoch % (epochs // 10) == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
    print("✅ GAN training complete.")

def generate_and_augment(X_train, y_train, generator, num_synthetic_multiplier=0.5):
    attack_data = X_train[y_train == 1].values
    if attack_data.shape[0] == 0:
        print("⚠️ No attack samples found. Skipping augmentation.")
        return X_train, y_train

    num_synthetic_samples = max(1, int(attack_data.shape[0] * num_synthetic_multiplier))
    noise = np.random.normal(0, 1, (num_synthetic_samples, generator.input_shape[1]))
    synthetic_attacks = generator.predict(noise, verbose=0)

    # Optional: Clip synthetic data to match scaled distribution better
    synthetic_attacks = np.clip(synthetic_attacks, -3, 3)

    synthetic_df = pd.DataFrame(synthetic_attacks, columns=X_train.columns)
    synthetic_labels = pd.Series(1, index=synthetic_df.index)

    X_aug = pd.concat([X_train, synthetic_df], ignore_index=True)
    y_aug = pd.concat([y_train, synthetic_labels], ignore_index=True)

    print(f"✅ Generated {num_synthetic_samples} synthetic attack samples.")
    return X_aug, y_aug

if __name__ == "__main__":
    data_dir = 'data/processed'
    os.makedirs('trained_models', exist_ok=True)

    print("🔹 Loading preprocessed UNSW-NB15 data...")
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train_unsw.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train_unsw.csv')).iloc[:, 0]  # Force Series

    latent_dim = 100
    gan_epochs = 5000
    batch_size = 64

    print("\n--- Training GAN for UNSW-NB15 ---")
    num_features = X_train.shape[1]
    generator = build_generator(latent_dim, num_features)
    discriminator = build_discriminator(num_features)
    gan = build_gan(generator, discriminator)

    attack_data = X_train[y_train == 1].values

    if attack_data.shape[0] > 100:
        train_gan(generator, discriminator, gan, attack_data,
                  epochs=gan_epochs, batch_size=batch_size, latent_dim=latent_dim)

        X_aug, y_aug = generate_and_augment(X_train, y_train, generator, num_synthetic_multiplier=0.5)

        # Save GAN models
        generator.save(os.path.join('trained_models', 'generator_unsw.h5'))
        discriminator.save(os.path.join('trained_models', 'discriminator_unsw.h5'))

        # Save augmented data
        X_aug.to_csv(os.path.join(data_dir, 'X_train_unsw_augmented.csv'), index=False)
        y_aug.to_csv(os.path.join(data_dir, 'y_train_unsw_augmented.csv'), index=False)
        print("✅ Augmented UNSW-NB15 data saved.")
    else:
        print(f"❌ Not enough attack samples ({attack_data.shape[0]}) to train GAN. Skipping augmentation.")
