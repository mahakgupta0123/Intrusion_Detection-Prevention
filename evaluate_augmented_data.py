import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
processed_data_dir = 'data/processed'
trained_models_dir = 'trained_models'
results_dir = 'results/evaluation_plots'  # Directory for plots and summary
os.makedirs(results_dir, exist_ok=True)

def train_and_evaluate_augmented(X_train_path, y_train_path, X_test_path, y_test_path, model_save_name, dataset_name):
    """
    Train and evaluate on augmented data.
    """
    print(f"\n--- Training IDS for {dataset_name} (Augmented Data) ---")

    # Load data
    X_train = pd.read_csv(os.path.join(processed_data_dir, X_train_path))
    y_train = pd.read_csv(os.path.join(processed_data_dir, y_train_path)).squeeze()
    X_test = pd.read_csv(os.path.join(processed_data_dir, X_test_path))
    y_test = pd.read_csv(os.path.join(processed_data_dir, y_test_path)).squeeze()

    print(f"Augmented Training Data Shape: {X_train.shape}, {y_train.shape}")
    print(f"Test Data Shape: {X_test.shape}, {y_test.shape}")

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, os.path.join(trained_models_dir, model_save_name))

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title(f'Confusion Matrix for {dataset_name} (Augmented Data)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'confusion_matrix_{dataset_name}_augmented.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Confusion matrix saved to {plot_path}")

    return {
        'dataset': dataset_name,
        'training_data': 'Augmented',
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': roc
    }

if __name__ == "__main__":
    results = []

    # --- Evaluate UNSW-NB15 (Augmented) ---
    results.append(train_and_evaluate_augmented(
        'X_train_unsw_augmented.csv', 'y_train_unsw_augmented.csv',
        'X_test_unsw.csv', 'y_test_unsw.csv',
        'ids_model_unsw_augmented.pkl', 'UNSW-NB15'
    ))

    # # --- Evaluate CICIDS-17 (Augmented) ---
    # results.append(train_and_evaluate_augmented(
    #     'X_train_cicids_augmented.csv', 'y_train_cicids_augmented.csv',
    #     'X_test_cicids.csv', 'y_test_cicids.csv',
    #     'ids_model_cicids_augmented.pkl', 'CICIDS-17'
    # ))

    # Save summary
    df_results = pd.DataFrame(results)
    summary_path = os.path.join(results_dir, 'ids_performance_augmented_data_summary.csv')
    df_results.to_csv(summary_path, index=False)
    print(f"\n✅ Augmented data evaluation summary saved to:\n{summary_path}")
