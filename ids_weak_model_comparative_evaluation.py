import pandas as pd
import numpy as np
import os
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
processed_data_dir = 'data/processed'
trained_models_dir = 'trained_models'
results_dir = 'results/evaluation_plots_weak_models'
summary_results_path = 'results/weak_model_performance_summary_unsw.xlsx'

# Ensure directories exist
os.makedirs(trained_models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

def train_and_evaluate_classifier(X_train_path, y_train_path, X_test_path, y_test_path,
                                  classifier_name, dataset_name, data_type):
    print(f"\n--- Training {classifier_name} for {dataset_name} ({data_type} Data) ---")

    # Load data
    X_train = pd.read_csv(os.path.join(processed_data_dir, X_train_path))
    y_train = pd.read_csv(os.path.join(processed_data_dir, y_train_path)).squeeze()

    X_test = pd.read_csv(os.path.join(processed_data_dir, X_test_path))
    y_test = pd.read_csv(os.path.join(processed_data_dir, y_test_path)).squeeze()

    # Drop unnecessary columns if present
    for df in [X_train, X_test]:
        for col in ['id', 'attack_cat']:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    print(f"Training Data Shape (X): {X_train.shape}, (y): {y_train.shape}")
    print(f"Testing Data Shape (X): {X_test.shape}, (y): {y_test.shape}")

    # Initialize model
    if classifier_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=3)
    elif classifier_name == "Logistic Regression":
        model = LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear', max_iter=1000)
    else:
        raise ValueError(f"Classifier '{classifier_name}' not supported.")

    # Train model
    model.fit(X_train, y_train)
    print(f"{classifier_name} training complete.")

    # Save model
    model_filename = f'ids_model_{dataset_name}_{classifier_name.lower().replace(" ", "_")}_{data_type.lower()}.pkl'
    model_path = os.path.join(trained_models_dir, model_filename)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title(f'{classifier_name} Confusion Matrix - {data_type} Data')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_plot_path = os.path.join(results_dir, f'confusion_matrix_{dataset_name}_{classifier_name.lower().replace(" ", "_")}_{data_type.lower()}.png')
    plt.savefig(cm_plot_path)
    plt.close()
    print(f"Confusion Matrix saved to {cm_plot_path}")

    return {
        'dataset': dataset_name,
        'classifier': classifier_name,
        'training_data_type': data_type,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

if __name__ == "__main__":
    all_results = []
    dataset_name = 'UNSW-NB15'
    weak_classifiers = ["Decision Tree", "Logistic Regression"]

    unsw_data_files = {
        'original': {'X_train': 'X_train_unsw.csv', 'y_train': 'y_train_unsw.csv'},
        'augmented': {'X_train': 'X_train_unsw_augmented.csv', 'y_train': 'y_train_unsw_augmented.csv'}
    }

    unsw_test_files = {'X_test': 'X_test_unsw.csv', 'y_test': 'y_test_unsw.csv'}

    for clf in weak_classifiers:
        # Original
        result_orig = train_and_evaluate_classifier(
            X_train_path=unsw_data_files['original']['X_train'],
            y_train_path=unsw_data_files['original']['y_train'],
            X_test_path=unsw_test_files['X_test'],
            y_test_path=unsw_test_files['y_test'],
            classifier_name=clf,
            dataset_name=dataset_name,
            data_type='Original'
        )
        all_results.append(result_orig)

        print("\n" + "-" * 50 + "\n")

        # Augmented
        result_aug = train_and_evaluate_classifier(
            X_train_path=unsw_data_files['augmented']['X_train'],
            y_train_path=unsw_data_files['augmented']['y_train'],
            X_test_path=unsw_test_files['X_test'],
            y_test_path=unsw_test_files['y_test'],
            classifier_name=clf,
            dataset_name=dataset_name,
            data_type='Augmented'
        )
        all_results.append(result_aug)

        print("\n" + "=" * 80 + "\n")

    # Save to Excel
    df_results = pd.DataFrame(all_results)
    with pd.ExcelWriter(summary_results_path, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='UNSW_WeakModels', index=False)

    print(f"\n✅ All results saved to {summary_results_path}")
    print("\n📊 Final Evaluation Summary:")
    print(df_results.round(4).to_string(index=False))
