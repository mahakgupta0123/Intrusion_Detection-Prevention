"""
Data Preprocessing Module

Handles loading, cleaning, and preprocessing of UNSW-NB15 dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import joblib
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Unified data preprocessing interface"""
    
    def __init__(self, data_dir='data/processed', raw_data_dir='data/UNSW‑NB15'):
        self.data_dir = data_dir
        self.raw_data_dir = raw_data_dir
        self.scaler = None
        self.label_encoders = {}
        
    def load_unsw_nb15(self):
        """Loads UNSW-NB15 training and testing files"""
        train_file = os.path.join(self.raw_data_dir, 'UNSW_NB15_training-set.csv')
        test_file = os.path.join(self.raw_data_dir, 'UNSW_NB15_testing-set.csv')

        if not os.path.exists(train_file) or not os.path.exists(test_file):
            raise FileNotFoundError(f"Missing dataset files. Check: {self.raw_data_dir}")

        logger.info(f"Loading training data from: {train_file}")
        df_train = pd.read_csv(train_file, low_memory=False)

        logger.info(f"Loading testing data from: {test_file}")
        df_test = pd.read_csv(test_file, low_memory=False)

        # Binarize labels: normal (0) vs attack (1)
        df_train['label'] = df_train['label'].apply(lambda x: 0 if x == 0 else 1)
        df_test['label'] = df_test['label'].apply(lambda x: 0 if x == 0 else 1)

        logger.info(f"✅ Loaded training: {df_train.shape}, testing: {df_test.shape}")
        return df_train, df_test

    def preprocess_data(self, df, fit_scaler=False):
        """
        Handles dropping, encoding, and scaling.
        
        Args:
            df: Input DataFrame
            fit_scaler: If True, fit new scaler. If False, use existing.
        """
        df = df.copy()
        df.columns = df.columns.str.strip()

        # Drop unnecessary columns
        for col in ['id', 'attack_cat']:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # Replace inf values with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Drop columns with >30% missing values
        df.dropna(axis=1, thresh=0.7 * len(df), inplace=True)

        # Impute missing values
        for col in df.select_dtypes(include=np.number).columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)

        for col in df.select_dtypes(include='object').columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)

        # Identify categorical columns
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include='object').columns.tolist()

        if 'label' in numerical_cols:
            numerical_cols.remove('label')
        if 'label' in categorical_cols:
            categorical_cols.remove('label')

        # Encode categorical features
        for col in categorical_cols:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                known = set(le.classes_)
                unseen = set(df[col].unique()) - known
                if unseen:
                    logger.warning(f"Column '{col}': unseen labels {unseen} → mapped to '{le.classes_[0]}'")
                    df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
                df[col] = le.transform(df[col])
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                if fit_scaler:
                    self.label_encoders[col] = le

        # Scale numerical features
        if fit_scaler:
            self.scaler = StandardScaler()
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Set fit_scaler=True on training data first.")
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])

        logger.info(f"✅ Preprocessed data shape: {df.shape}")
        return df

    def process_pipeline(self):
        """Complete preprocessing pipeline"""
        os.makedirs(self.data_dir, exist_ok=True)

        logger.info("🔄 Loading UNSW-NB15...")
        df_train_raw, df_test_raw = self.load_unsw_nb15()

        logger.info("🔄 Preprocessing training data...")
        df_train_processed = self.preprocess_data(df_train_raw, fit_scaler=True)
        X_train = df_train_processed.drop('label', axis=1)
        y_train = df_train_processed['label']

        logger.info("🔄 Preprocessing test data...")
        y_test = df_test_raw['label'].copy()  # extract label before preprocessing
        df_test_processed = self.preprocess_data(df_test_raw, fit_scaler=False)

        # Drop label from test processed df if it survived
        if 'label' in df_test_processed.columns:
            df_test_processed.drop(columns=['label'], inplace=True)

        # Align columns to match training features exactly
        for col in set(X_train.columns) - set(df_test_processed.columns):
            df_test_processed[col] = 0
        for col in set(df_test_processed.columns) - set(X_train.columns):
            df_test_processed.drop(columns=[col], inplace=True)

        X_test = df_test_processed[X_train.columns]  # enforce column order
        
        # Align column order
        X_test = X_test[X_train.columns]

        logger.info(f"Final shapes → Train: {X_train.shape}, Test: {X_test.shape}")
        logger.info(f"Train distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"Test distribution: {y_test.value_counts().to_dict()}")

        # Save processed data
        X_train.to_csv(os.path.join(self.data_dir, 'X_train_unsw.csv'), index=False)
        X_test.to_csv(os.path.join(self.data_dir, 'X_test_unsw.csv'), index=False)
        y_train.to_csv(os.path.join(self.data_dir, 'y_train_unsw.csv'), index=False)
        y_test.to_csv(os.path.join(self.data_dir, 'y_test_unsw.csv'), index=False)
        joblib.dump(self.scaler, os.path.join(self.data_dir, 'unsw_scaler.pkl'))
        joblib.dump(self.label_encoders, os.path.join(self.data_dir, 'label_encoders.pkl'))

        logger.info("✅ All preprocessed data saved.")
        return X_train, y_train, X_test, y_test


# Backward compatibility functions
def load_unsw_nb15():
    """Load UNSW-NB15 dataset"""
    preprocessor = DataPreprocessor()
    return preprocessor.load_unsw_nb15()


def preprocess_data(df, scaler=None, label_encoders=None):
    """Preprocess data (legacy function)"""
    preprocessor = DataPreprocessor()
    preprocessor.scaler = scaler
    preprocessor.label_encoders = label_encoders or {}
    return preprocessor.preprocess_data(df, fit_scaler=scaler is None)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    preprocessor = DataPreprocessor(
        data_dir='../../data/processed',
        raw_data_dir='../../data/UNSW‑NB15'
    )
    X_train, y_train, X_test, y_test = preprocessor.process_pipeline()
