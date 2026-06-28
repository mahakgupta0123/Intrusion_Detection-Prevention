"""
Model Evaluation Module

Proper evaluation with stratified k-fold cross-validation for imbalanced datasets.
"""

import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, make_scorer
)
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Model evaluation with stratified k-fold"""
    
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv_results = None
        self.test_metrics = None
    
    def stratified_kfold_cv(self, X_train, y_train, X_test, y_test, 
                            model_type='rf', verbose=True):
        """
        Perform stratified k-fold cross-validation
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            model_type: 'rf' or 'xgb'
            verbose: Print details
        
        Returns:
            results_dict with CV metrics and final test evaluation
        """
        
        print(f"\n{'='*70}")
        print(f"🔄 Stratified {self.n_splits}-Fold Cross-Validation")
        print(f"{'='*70}")
        
        # Analyze class distribution
        n_normal = (y_train == 0).sum()
        n_attacks = (y_train == 1).sum()
        imbalance_ratio = n_normal / n_attacks
        
        logger.info(f"\n📊 Training Data Distribution:")
        logger.info(f"   Normal: {n_normal} ({100*n_normal/len(y_train):.1f}%)")
        logger.info(f"   Attack: {n_attacks} ({100*n_attacks/len(y_train):.1f}%)")
        logger.info(f"   Imbalance ratio: {imbalance_ratio:.2f}x")
        
        # Create model
        if model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=200, random_state=self.random_state,
                class_weight='balanced', n_jobs=-1, max_depth=15
            )
        elif model_type == 'xgb':
            try:
                from xgboost import XGBClassifier
                scale_pos_weight = n_normal / n_attacks
                model = XGBClassifier(
                    n_estimators=200, random_state=self.random_state,
                    scale_pos_weight=scale_pos_weight,
                    max_depth=6, learning_rate=0.05, n_jobs=-1, eval_metric='logloss'
                )
            except ImportError:
                logger.error("XGBoost not installed")
                return None
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Define scoring metrics
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score, zero_division=0),
            'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0),
            'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
            'roc_auc': make_scorer(roc_auc_score)
        }
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        cv_results = cross_validate(model, X_train, y_train, cv=skf, scoring=scoring,
                                    return_train_score=True, n_jobs=-1)
        
        self.cv_results = cv_results
        
        # Print CV results
        logger.info(f"\n📈 Cross-Validation Results ({self.n_splits} folds):")
        logger.info(f"{'Metric':<20} {'Mean CV':<12} {'Std Dev':<12} {'Train':<12}")
        logger.info("-" * 56)
        
        for metric in ['accuracy', 'precision', 'recall', 'f1_weighted', 'f1_macro', 'roc_auc']:
            cv_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            mean_cv = np.mean(cv_scores)
            std_cv = np.std(cv_scores)
            mean_train = np.mean(train_scores)
            
            logger.info(f"{metric:<20} {mean_cv:<12.4f} {std_cv:<12.4f} {mean_train:<12.4f}")
            
            if mean_train - mean_cv > 0.1:
                logger.warning(f"   ⚠️  Possible overfitting (gap: {mean_train - mean_cv:.4f})")
        
        # Train final model and evaluate on test set
        logger.info(f"\n🔧 Training final model on all training data...")
        model.fit(X_train, y_train)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"📋 Final Test Set Evaluation")
        logger.info(f"{'='*70}")
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        self.test_metrics = test_metrics
        
        logger.info(f"\n{'Metric':<20} {'Score':<12}")
        logger.info("-" * 32)
        for metric, value in test_metrics.items():
            logger.info(f"{metric:<20} {value:<12.4f}")
        
        # Confusion matrix analysis
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        logger.info(f"\n🎯 Confusion Matrix Breakdown:")
        logger.info(f"   TN: {tn}, FP: {fp}")
        logger.info(f"   FN: {fn}, TP: {tp}")
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        logger.info(f"   FPR: {fpr:.4f}, FNR: {fnr:.4f}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Attack'],
                    yticklabels=['Normal', 'Attack'],
                    cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {model_type.upper()} ({self.n_splits}-Fold CV)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/confusion_matrix_{model_type}_cv.png', dpi=300)
        logger.info(f"\n✅ Confusion matrix saved")
        plt.close()
        
        # Classification report
        logger.info(f"\n📄 Classification Report:")
        logger.info(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
        
        return {
            'model': model,
            'cv_results': cv_results,
            'test_metrics': test_metrics,
            'y_pred': y_pred,
            'y_proba': y_proba
        }


def stratified_kfold_evaluation(X_train, y_train, X_test, y_test, 
                                model_type='rf', n_splits=5, verbose=True):
    """Standalone function for stratified k-fold evaluation"""
    evaluator = ModelEvaluator(n_splits=n_splits)
    return evaluator.stratified_kfold_cv(X_train, y_train, X_test, y_test, 
                                         model_type=model_type, verbose=verbose)


def compare_models_with_cv(X_train, y_train, X_test, y_test):
    """Compare multiple models"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    models_config = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1
        ),
        'LogisticRegression': LogisticRegression(
            random_state=42, class_weight='balanced', max_iter=1000
        ),
    }
    
    try:
        from xgboost import XGBClassifier
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        models_config['XGBoost'] = XGBClassifier(
            n_estimators=200, random_state=42,
            scale_pos_weight=scale_pos_weight,
            max_depth=6, learning_rate=0.05, n_jobs=-1, eval_metric='logloss'
        )
    except ImportError:
        pass
    
    results_summary = []
    
    for model_name in models_config.keys():
        logger.info(f"\n\n{'#'*70}\n# {model_name}\n{'#'*70}")
        
        try:
            evaluator = ModelEvaluator(n_splits=5)
            result = evaluator.stratified_kfold_cv(
                X_train, y_train, X_test, y_test,
                model_type=model_name.lower()
            )
            
            results_summary.append({
                'Model': model_name,
                **result['test_metrics']
            })
        except Exception as e:
            logger.error(f"Error: {e}")
    
    df_comparison = pd.DataFrame(results_summary)
    logger.info(f"\n\n{'='*70}\n📊 Model Comparison\n{'='*70}")
    logger.info(df_comparison.to_string(index=False))
    
    os.makedirs('results', exist_ok=True)
    df_comparison.to_csv('results/model_comparison_cv.csv', index=False)
    logger.info(f"\n✅ Comparison saved to results/model_comparison_cv.csv")
    
    return df_comparison


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Model evaluator module loaded")
