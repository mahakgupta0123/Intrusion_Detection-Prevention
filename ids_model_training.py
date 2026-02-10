import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate_ids(X_train_augmented, y_train_augmented, X_test, y_test, dataset_name):
    print(f"\n--- Training IDS for {dataset_name} ---")
    print(f"Augmented Training Data Shape (X): {X_train_augmented.shape}, (y): {y_train_augmented.shape}")
    print(f"Original Testing Data Shape (X): {X_test.shape}, (y): {y_test.shape}")

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    print(f"Training {dataset_name} Random Forest Classifier...")
    model.fit(X_train_augmented, y_train_augmented)
    print(f"{dataset_name} IDS Model Training Complete.")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n--- Evaluating IDS for {dataset_name} on Test Set ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall (Detection Rate): {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

    if len(np.unique(y_test)) > 1:
        try:
            print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
        except ValueError:
            print("ROC AUC score could not be calculated.")
    else:
        print("ROC AUC score not applicable (only one class present in y_test).")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal (0)', 'Attack (1)'], yticklabels=['Normal (0)', 'Attack (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {dataset_name} IDS')

    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, f'confusion_matrix_{dataset_name}.png'))
    plt.close()

    return model

if __name__ == "__main__":
    processed_data_dir = 'data/processed'
    trained_models_dir = 'trained_models'
    os.makedirs(trained_models_dir, exist_ok=True)

    print("🔄 Loading UNSW-NB15 augmented training and original testing data...")

    try:
        X_train = pd.read_csv(os.path.join(processed_data_dir, 'X_train_unsw_augmented.csv'))
        y_train = pd.read_csv(os.path.join(processed_data_dir, 'y_train_unsw_augmented.csv')).squeeze()
    except FileNotFoundError:
        print("⚠️ Augmented UNSW-NB15 data not found. Falling back to original training set.")
        X_train = pd.read_csv(os.path.join(processed_data_dir, 'X_train_unsw.csv'))
        y_train = pd.read_csv(os.path.join(processed_data_dir, 'y_train_unsw.csv')).squeeze()

    X_test = pd.read_csv(os.path.join(processed_data_dir, 'X_test_unsw.csv'))
    y_test = pd.read_csv(os.path.join(processed_data_dir, 'y_test_unsw.csv')).squeeze()

    model = train_and_evaluate_ids(X_train, y_train, X_test, y_test, "UNSW-NB15")
    joblib.dump(model, os.path.join(trained_models_dir, 'ids_model_unsw.pkl'))
    print("✅ UNSW-NB15 IDS model saved.")
