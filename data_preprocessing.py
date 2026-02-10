
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import joblib

def load_unsw_nb15():
    """Loads UNSW-NB15 training and testing files from an absolute path."""
    data_dir = r"/home/mahak/IPDRS_research/data/UNSW‑NB15"

    train_file = os.path.join(data_dir, 'UNSW_NB15_training-set.csv')
    test_file = os.path.join(data_dir, 'UNSW_NB15_testing-set.csv')

    print("🔍 Looking for training file at:", train_file)
    print("🔍 Looking for testing file at:", test_file)

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("❌ Error: Training or Testing file missing.")
        return None, None

    print(f"📥 Loading training data from: {train_file}")
    df_train = pd.read_csv(train_file, low_memory=False)

    print(f"📥 Loading testing data from: {test_file}")
    df_test = pd.read_csv(test_file, low_memory=False)

    df_train['label'] = df_train['label'].apply(lambda x: 0 if x == 0 else 1)
    df_test['label'] = df_test['label'].apply(lambda x: 0 if x == 0 else 1)

    return df_train, df_test

def preprocess_data(df):
    """Handles dropping, encoding, and scaling."""
    df.columns = df.columns.str.strip()

    for col in ['id', 'attack_cat']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=1, thresh=0.7 * len(df), inplace=True)

    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    for col in df.select_dtypes(include='object').columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    if 'label' in numerical_cols:
        numerical_cols.remove('label')
    if 'label' in categorical_cols:
        categorical_cols.remove('label')

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df, scaler, label_encoders

if __name__ == "__main__":
    processed_data_dir = 'data/processed'
    os.makedirs(processed_data_dir, exist_ok=True)

    print("--- Loading UNSW-NB15 ---")
    df_train_raw, df_test_raw = load_unsw_nb15()
    if df_train_raw is None or df_test_raw is None:
        exit()

    print("\n--- Preprocessing UNSW-NB15 Training Data ---")
    df_train_processed, scaler, label_encoders = preprocess_data(df_train_raw.copy())
    X_train = df_train_processed.drop('label', axis=1)
    y_train = df_train_processed['label']

    print("\n--- Preprocessing UNSW-NB15 Testing Data ---")
    df_test_processed = df_test_raw.copy()
    df_test_processed.columns = df_test_processed.columns.str.strip()
    df_test_processed.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop any columns missing in processed train set
    cols_to_drop = [col for col in df_test_processed.columns if col not in df_train_processed.columns]
    df_test_processed.drop(columns=cols_to_drop, errors='ignore', inplace=True)

    # Impute and encode test set
    for col in df_test_processed.select_dtypes(include=np.number).columns:
        if col != 'label' and df_test_processed[col].isnull().any():
            median_val = X_train[col].median() if col in X_train.columns else df_test_processed[col].median()
            df_test_processed[col].fillna(median_val, inplace=True)

    for col in df_test_processed.select_dtypes(include='object').columns:
        if col in label_encoders:
            df_test_processed[col] = df_test_processed[col].map(
                lambda s: s if s in label_encoders[col].classes_ else label_encoders[col].classes_[0])
            df_test_processed[col] = label_encoders[col].transform(df_test_processed[col])
        else:
            le = LabelEncoder()
            df_test_processed[col] = le.fit_transform(df_test_processed[col])

    numerical_cols = [col for col in df_test_processed.columns if col in scaler.feature_names_in_ and col != 'label']
    df_test_processed[numerical_cols] = scaler.transform(df_test_processed[numerical_cols])

    X_test = df_test_processed.drop('label', axis=1)
    y_test = df_test_processed['label']

    # Align columns
    for col in set(X_train.columns) - set(X_test.columns):
        X_test[col] = 0
    for col in set(X_test.columns) - set(X_train.columns):
        X_train[col] = 0
    X_test = X_test[X_train.columns]

    print(f"\nUNSW-NB15 Final Shapes → Train: {X_train.shape}, Test: {X_test.shape}")

    # Save
    X_train.to_csv(os.path.join(processed_data_dir, 'X_train_unsw.csv'), index=False)
    X_test.to_csv(os.path.join(processed_data_dir, 'X_test_unsw.csv'), index=False)
    y_train.to_csv(os.path.join(processed_data_dir, 'y_train_unsw.csv'), index=False)
    y_test.to_csv(os.path.join(processed_data_dir, 'y_test_unsw.csv'), index=False)

    joblib.dump(scaler, os.path.join(processed_data_dir, 'unsw_scaler.pkl'))
    joblib.dump(label_encoders, os.path.join(processed_data_dir, 'unsw_label_encoders.pkl'))

    print("\n✅ Preprocessing complete. Processed files saved.")