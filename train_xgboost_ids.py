import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate_xgb(X_train, y_train, X_test, y_test, dataset_name="UNSW-NB15"):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n--- XGBoost IDS Evaluation for {dataset_name} ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title(f'Confusion Matrix - XGBoost {dataset_name}')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/confusion_matrix_xgboost_{dataset_name}.png')
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

    model = train_and_evaluate_xgb(X_train, y_train, X_test, y_test)
    joblib.dump(model, os.path.join(model_dir, 'xgb_model_unsw.pkl'))
    print("✅ XGBoost model saved.")
