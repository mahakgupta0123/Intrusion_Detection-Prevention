"""
Neural Network Models (LSTM, CNN)

Deep learning models for IDS tasks.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import logging

logger = logging.getLogger(__name__)


def build_lstm_model(input_shape, units=64, dropout=0.3):
    """Build LSTM model for sequence classification"""
    model = models.Sequential([
        layers.LSTM(units, activation='relu', return_sequences=True, input_shape=input_shape),
        layers.Dropout(dropout),
        layers.LSTM(units // 2, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def build_cnn_model(input_shape, num_classes=2):
    """Build CNN model for classification"""
    model = models.Sequential([
        layers.Conv1D(64, 3, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(32, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def train_lstm_model(X_train, y_train, X_test=None, y_test=None, epochs=50, batch_size=32):
    """Train LSTM model"""
    logger.info(f"🔄 Building LSTM model...")
    
    # Reshape for LSTM (samples, timesteps, features)
    if len(X_train.shape) == 2:
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        if X_test is not None:
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    model = build_lstm_model(X_train.shape[1:])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    logger.info(f"Training LSTM for {epochs} epochs...")
    
    early_stop = callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test) if X_test is not None else None,
        callbacks=[early_stop],
        verbose=1
    )
    
    logger.info("✅ LSTM training complete")
    return model, history


def train_cnn_model(X_train, y_train, X_test=None, y_test=None, epochs=50, batch_size=32):
    """Train CNN model"""
    logger.info(f"🔄 Building CNN model...")
    
    # Reshape for CNN
    if len(X_train.shape) == 2:
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        if X_test is not None:
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    model = build_cnn_model(X_train.shape[1:])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    logger.info(f"Training CNN for {epochs} epochs...")
    
    early_stop = callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test) if X_test is not None else None,
        callbacks=[early_stop],
        verbose=1
    )
    
    logger.info("✅ CNN training complete")
    return model, history


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Neural network models module loaded")
