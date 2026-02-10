import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate_lstm(X_train, y_train, X_test, y_test, dataset_name="UNSW-NB15"):
    # Reshape to 3D [samples, timesteps, features]
    X_train = np.expand_dims(X_train.values, axis=1)
    X_test = np.expand_dims(X_test.values, axis=1)

    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1)

    y_proba = model.predict(X_test).ravel()
    y_pred = (y_proba > 0.5).astype(int)

    print(f"\n--- LSTM IDS Evaluation for {dataset_name} ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title(f'Confusion Matrix - LSTM {dataset_name}')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/confusion_matrix_lstm_{dataset_name}.png')
    plt.close()

    return model

if __name__ == "__main__":
    data_dir = 'data/processed'
    model_dir = 'trained_models'
    os.makedirs(model_dir, exist_ok=True)

    print("🔹 Loading data...")
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train_unsw_augmented.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train_unsw_augmented.csv')).squeeze()
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test_unsw.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test_unsw.csv')).squeeze()

    model = train_and_evaluate_lstm(X_train, y_train, X_test, y_test)
    model.save(os.path.join(model_dir, 'lstm_model_unsw.h5'))
    print("✅ LSTM model saved.")
