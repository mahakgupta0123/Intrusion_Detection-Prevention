import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
processed_data_dir = 'data/processed'
trained_models_dir = 'trained_models'
results_dir = 'results/evaluation_plots'  # Directory for confusion matrix plots

# Ensure directories exist
os.makedirs(trained_models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

def train_and_evaluate_ids(X_train_path, y_train_path, X_test_path, y_test_path, model_save_name, dataset_name):
    print(f"\n--- Training IDS for {dataset_name} (Original Data) ---")

    # Load data
    X_train = pd.read_csv(os.path.join(processed_data_dir, X_train_path))
    y_train = pd.read_csv(os.path.join(processed_data_dir, y_train_path)).squeeze()

    X_test = pd.read_csv(os.path.join(processed_data_dir, X_test_path))
    y_test = pd.read_csv(os.path.join(processed_data_dir, y_test_path)).squeeze()

    # Drop unnecessary columns if they exist
    for df in [X_train, X_test]:
        for col in ['id', 'attack_cat']:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    print("\nClass Distribution in Training Set:")
    print(y_train.value_counts())  # Show class imbalance

    print(f"\nOriginal Training Data Shape (X): {X_train.shape}, (y): {y_train.shape}")
    print(f"Original Testing Data Shape (X): {X_test.shape}, (y): {y_test.shape}")

    # Train Random Forest with class weight balancing
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # ✅ Helps with imbalance in original data
    )
    model.fit(X_train, y_train)
    print(f"{dataset_name} IDS Model Training Complete.")

    # Save model
    model_path = os.path.join(trained_models_dir, model_save_name)
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\n--- Evaluation Metrics for {dataset_name} ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title(f'Confusion Matrix - {dataset_name} (Original Data)')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_path = os.path.join(results_dir, f'confusion_matrix_{dataset_name}_original.png')
    plt.savefig(cm_path)
    print(f"Confusion Matrix saved to: {cm_path}")
    plt.close()

    return {
        'dataset': dataset_name,
        'training_data': 'Original',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

if __name__ == "__main__":
    # Only UNSW-NB15 evaluation
    results = []
    unsw_results = train_and_evaluate_ids(
        X_train_path='X_train_unsw.csv',
        y_train_path='y_train_unsw.csv',
        X_test_path='X_test_unsw.csv',
        y_test_path='y_test_unsw.csv',
        model_save_name='rf_model_unsw_original.pkl',
        dataset_name='UNSW-NB15'
    )
    results.append(unsw_results)

    # Save summary
    df_results = pd.DataFrame(results)
    print("\nSummary of IDS Evaluation on Original Data:")
    print(df_results.round(4).to_string(index=False))

    summary_path = os.path.join(results_dir, 'rf_evaluation_original_summary.csv')
    df_results.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")
