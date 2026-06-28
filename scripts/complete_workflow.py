#!/usr/bin/env python3
"""
Complete End-to-End IDS/IPS Pipeline with Augmentation, Detection, Prevention, and XAI

This script demonstrates the complete workflow:
1. Data Preprocessing
2. Data Augmentation (GAN + SMOTE)
3. Feature Extraction & Selection
4. Model Training (IDS Detection)
5. Model Evaluation
6. Explainability Analysis (SHAP/LIME)
7. Prevention Model Training
8. Real-time IPS Simulation

Usage:
    python scripts/complete_workflow.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import DataPreprocessor
from src.augmentation import GANAugmentor, combine_gan_and_smote
from src.features import FeatureExtractor
from src.models import ModelTrainer
from src.evaluation import stratified_kfold_evaluation, compare_models_with_cv
from src.xai import SHAPExplainer, generate_shap_csv, LIMEExplainer
from src.prevention import PreventionModel, apply_prevention_rules
from src.core import PredictionEngine, RealTimeIPS
from src.utils import MetricsLogger, ensure_directory, plot_confusion_matrix


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def main():
    """Execute complete IDS/IPS pipeline."""
    
    start_time = datetime.now()
    print("\n" + "🚀 "*40)
    print("COMPLETE END-TO-END IDS/IPS PIPELINE")
    print("🚀 "*40 + "\n")
    
    # =========================================================================
    # STEP 1: DATA PREPROCESSING
    # =========================================================================
    print_section("STEP 1: DATA PREPROCESSING")
    
    print("📊 Loading and preprocessing UNSW-NB15 dataset...")
    preprocessor = DataPreprocessor(
        data_dir='data/processed',
        raw_data_dir='data/UNSW‑NB15'
    )
    
    X_train, y_train, X_test, y_test = preprocessor.process_pipeline()
    
    print(f"✅ Training set: {X_train.shape}")
    print(f"✅ Test set: {X_test.shape}")
    print(f"✅ Training class distribution:")
    print(f"   Normal: {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
    print(f"   Attack: {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")
    
    # =========================================================================
    # STEP 2: FEATURE EXTRACTION & ANALYSIS
    # =========================================================================
    print_section("STEP 2: FEATURE EXTRACTION & ANALYSIS")
    
    print("🔍 Analyzing features...")
    print(f"   Total features: {X_train.shape[1]}")
    print(f"   Feature names: {list(X_train.columns[:5])}... (showing first 5)")
    
    # Feature statistics
    print(f"\n📈 Feature Statistics (Training Set):")
    print(f"   Mean feature value: {X_train.values.mean():.4f}")
    print(f"   Std feature value: {X_train.values.std():.4f}")
    print(f"   Min feature value: {X_train.values.min():.4f}")
    print(f"   Max feature value: {X_train.values.max():.4f}")
    
    # =========================================================================
    # STEP 3: DATA AUGMENTATION
    # =========================================================================
    print_section("STEP 3: DATA AUGMENTATION (GAN + SMOTE)")
    
    print("🤖 Training GAN on attack samples...")
    print(f"   Attack samples available: {(y_train == 1).sum()}")
    
    # Train GAN
    augmentor = GANAugmentor(
        latent_dim=100,
        batch_size=64,
        learning_rate=0.0002
    )
    
    attack_data = X_train[y_train == 1].values
    print(f"   Training GAN with {len(attack_data)} attack samples...")
    
    augmentor.train(
        attack_data,
        epochs=5000,
    )
    print("✅ GAN training complete")
    
    # Apply GAN augmentation
    print(f"\n🔄 Generating synthetic data (1.5x multiplier)...")
    X_augmented, y_augmented = augmentor.augment_data(
        X_train, y_train,
        multiplier=1.5
    )
    print(f"   Augmented dataset: {X_augmented.shape}")
    print(f"   New class distribution:")
    print(f"   Normal: {(y_augmented == 0).sum()} ({(y_augmented == 0).sum()/len(y_augmented)*100:.1f}%)")
    print(f"   Attack: {(y_augmented == 1).sum()} ({(y_augmented == 1).sum()/len(y_augmented)*100:.1f}%)")
    
    # Apply SMOTE
    print(f"\n✨ Applying SMOTE for optimal balance...")
    X_final, y_final = combine_gan_and_smote(
        X_augmented, y_augmented,
        apply_smote=True,
        smote_ratio=0.8
    )
    print(f"   Final dataset: {X_final.shape}")
    print(f"   Final class distribution:")
    print(f"   Normal: {(y_final == 0).sum()} ({(y_final == 0).sum()/len(y_final)*100:.1f}%)")
    print(f"   Attack: {(y_final == 1).sum()} ({(y_final == 1).sum()/len(y_final)*100:.1f}%)")
    
    # Save augmentor
    ensure_directory('trained_models')
    augmentor.save_models('trained_models')
    print("✅ GAN models saved")
    
    # =========================================================================
    # STEP 4: MODEL TRAINING (IDS Detection)
    # =========================================================================
    print_section("STEP 4: MODEL TRAINING (IDS Detection)")
    
    print("🧠 Training detection models...")
    
    trainer = ModelTrainer()
    
    # Train Random Forest
    print("\n1️⃣  Training Random Forest...")
    rf_model = trainer.train_random_forest(
        X_final, y_final,
        n_estimators=200,
        max_depth=15,
    )
    print("✅ Random Forest trained")
    
    # Train XGBoost
    print("\n2️⃣  Training XGBoost...")
    xgb_model = trainer.train_xgboost(
        X_final, y_final,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
    )
    print("✅ XGBoost trained")
    
    # Save models
    trainer.save('ids_model_rf_augmented.pkl')
    print("✅ Models saved to trained_models/")
    
    # =========================================================================
    # STEP 5: MODEL EVALUATION
    # =========================================================================
    print_section("STEP 5: MODEL EVALUATION (Stratified k-fold CV)")
    
    print("📊 Evaluating models with stratified 5-fold cross-validation...")
    
    # Evaluate Random Forest
    print("\n1️⃣  Random Forest Evaluation:")
    rf_results = stratified_kfold_evaluation(
        X_final, y_final,
        X_test, y_test,
        model_type='rf',
        n_splits=5
    )
    
    print(f"\n   Train Metrics:")
    for metric, value in rf_results['train_metrics'].items():
        print(f"     {metric}: {value:.4f}")
    
    print(f"\n   Test Metrics:")
    for metric, value in rf_results['test_metrics'].items():
        print(f"     {metric}: {value:.4f}")
    
    # Evaluate XGBoost
    print("\n2️⃣  XGBoost Evaluation:")
    xgb_results = stratified_kfold_evaluation(
        X_final, y_final,
        X_test, y_test,
        model_type='xgb',
        n_splits=5
    )
    
    print(f"\n   Train Metrics:")
    for metric, value in xgb_results['train_metrics'].items():
        print(f"     {metric}: {value:.4f}")
    
    print(f"\n   Test Metrics:")
    for metric, value in xgb_results['test_metrics'].items():
        print(f"     {metric}: {value:.4f}")
    
    # Model comparison
    print("\n📈 Model Comparison:")
    comparison_df = compare_models_with_cv(
        X_final, y_final,
        X_test, y_test
    )
    print(comparison_df.to_string())
    
    # Save results
    ensure_directory('results')
    comparison_df.to_csv('results/model_comparison_cv.csv', index=False)
    print("✅ Comparison saved to results/model_comparison_cv.csv")
    
    # =========================================================================
    # STEP 6: EXPLAINABILITY ANALYSIS (SHAP & LIME)
    # =========================================================================
    print_section("STEP 6: EXPLAINABILITY ANALYSIS (SHAP & LIME)")
    
    print("🔍 Generating SHAP explanations...")
    
    # SHAP Analysis
    ensure_directory('results/xai_explanations')
    
    print("   Generating SHAP values...")
    feature_importance_df = generate_shap_csv(
        rf_model,
        X_test,
        output_path='results/xai_explanations/shap_values.csv',
        model_type='tree'
    )
    
    print("\n   Top 10 Most Important Features (by SHAP):")
    print(feature_importance_df.head(10).to_string(index=False))
    
    # SHAP Visualizations
    print("\n   Generating SHAP visualizations...")
    shap_explainer = SHAPExplainer(rf_model, X_test, model_type='tree')
    shap_explainer.save_all_plots('results/xai_explanations')
    print("✅ SHAP plots saved:")
    print("   - shap_summary_beeswarm.png")
    print("   - shap_bar_plot.png")
    print("   - shap_force_plot_instance_0.html")
    
    # LIME Analysis
    print("\n🔍 Generating LIME explanations...")
    
    lime_explainer = LIMEExplainer(rf_model, X_final, feature_names=X_test.columns.tolist())
    print("   Explaining first test instance...")
    lime_exp = lime_explainer.explain_instance(X_test.iloc[0])
    lime_explainer.save_explanation_html(
        X_test.iloc[0],
        'results/xai_explanations/lime_explanation_instance_0.html'
    )
    print("✅ LIME explanation saved: lime_explanation_instance_0.html")
    
    # =========================================================================
    # STEP 7: FEATURE SELECTION (using SHAP)
    # =========================================================================
    print_section("STEP 7: FEATURE SELECTION (Top Features)")
    
    top_features = feature_importance_df['feature'].head(10).tolist()
    print(f"📌 Selected Top 10 Features:")
    for i, feat in enumerate(top_features, 1):
        print(f"   {i}. {feat}")
    
    # =========================================================================
    # STEP 8: PREVENTION MODEL TRAINING
    # =========================================================================
    print_section("STEP 8: PREVENTION MODEL TRAINING")
    
    print("🛡️  Training prevention model using top features...")
    
    prevention_model = PreventionModel(model_type='xgboost')
    prevention_model.train(
        X_final, y_final,
        top_features=top_features,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1
    )
    
    prevention_model.save('trained_models/prevention_model.pkl')
    print("✅ Prevention model trained and saved")
    
    # Test prevention rules
    print("\n⚙️  Applying prevention rules...")
    X_test_subset = X_test[top_features].copy()
    X_with_flags = apply_prevention_rules(X_test_subset, top_features)
    
    flagged_count = X_with_flags['prevent_flag'].sum()
    print(f"   Packets flagged for prevention: {flagged_count} / {len(X_test)}")
    print(f"   Prevention rate: {flagged_count/len(X_test)*100:.2f}%")
    
    # =========================================================================
    # STEP 9: REAL-TIME IPS SIMULATION
    # =========================================================================
    print_section("STEP 9: REAL-TIME IPS SIMULATION")
    
    print("🔌 Creating real-time IPS engine...")
    
    prediction_engine = PredictionEngine(
        'trained_models/ids_model_rf_augmented.pkl',
        'data/processed/unsw_scaler.pkl'
    )
    
    ips = RealTimeIPS(
        prediction_engine,
        alert_threshold=0.7
    )
    
    print("✅ IPS engine created")
    
    # Simulate real-time traffic analysis
    print("\n🚦 Simulating real-time traffic analysis...")
    
    # Analyze test set as traffic stream
    sample_size = min(100, len(X_test))
    X_sample = X_test.iloc[:sample_size]
    
    print(f"   Analyzing {sample_size} packets...")
    
    results_list = []
    alerts_generated = 0
    
    for idx in range(sample_size):
        packet_features = X_sample.iloc[idx].values
        is_attack, alert_info = ips.analyze_packet(packet_features)
        
        if alert_info:
            alerts_generated += 1
            results_list.append({
                'packet_id': idx,
                'is_attack': is_attack,
                'confidence': alert_info.get('confidence', 0),
                'alert_level': alert_info.get('alert_level', 'INFO')
            })
    
    print(f"✅ Alerts generated: {alerts_generated} / {sample_size}")
    
    # Get IPS summary
    summary = ips.get_alert_summary()
    print(f"\n📊 IPS Summary:")
    print(f"   Total packets analyzed: {summary['total_packets']}")
    print(f"   Total alerts: {summary['total_alerts']}")
    print(f"   Average confidence: {summary['avg_confidence']:.4f}")
    
    # =========================================================================
    # STEP 10: FINAL METRICS & REPORTING
    # =========================================================================
    print_section("STEP 10: FINAL METRICS & REPORTING")
    
    # Create comprehensive metrics report
    metrics_report = {
        'timestamp': datetime.now().isoformat(),
        'duration_seconds': (datetime.now() - start_time).total_seconds(),
        'preprocessing': {
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'train_class_distribution': {
                'normal': int((y_train == 0).sum()),
                'attack': int((y_train == 1).sum())
            }
        },
        'augmentation': {
            'original_samples': len(y_train),
            'augmented_samples': len(y_final),
            'augmentation_ratio': len(y_final) / len(y_train),
            'final_class_distribution': {
                'normal': int((y_final == 0).sum()),
                'attack': int((y_final == 1).sum())
            }
        },
        'models': {
            'random_forest': rf_results['test_metrics'],
            'xgboost': xgb_results['test_metrics']
        },
        'xai': {
            'top_10_features': top_features,
            'num_features': X_train.shape[1]
        },
        'prevention': {
            'prevention_rate': flagged_count / len(X_test),
            'flagged_packets': flagged_count,
            'total_packets': len(X_test)
        },
        'ips': {
            'packets_analyzed': summary['total_packets'],
            'alerts_generated': summary['total_alerts'],
            'avg_confidence': float(summary['avg_confidence'])
        }
    }
    
    print("📊 Pipeline Metrics:")
    print(f"   Total execution time: {metrics_report['duration_seconds']:.2f} seconds")
    print(f"\n   Augmentation:")
    print(f"     Original samples: {metrics_report['augmentation']['original_samples']}")
    print(f"     Final samples: {metrics_report['augmentation']['augmented_samples']}")
    print(f"     Multiplier: {metrics_report['augmentation']['augmentation_ratio']:.2f}x")
    
    print(f"\n   Best Model Performance (Test Set):")
    best_model = 'Random Forest' if rf_results['test_metrics']['f1_weighted'] >= xgb_results['test_metrics']['f1_weighted'] else 'XGBoost'
    best_metrics = rf_results['test_metrics'] if best_model == 'Random Forest' else xgb_results['test_metrics']
    
    print(f"     Model: {best_model}")
    print(f"     Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"     Recall: {best_metrics['recall']:.4f}")
    print(f"     Precision: {best_metrics['precision']:.4f}")
    print(f"     F1-Score: {best_metrics['f1_weighted']:.4f}")
    print(f"     ROC-AUC: {best_metrics['roc_auc']:.4f}")
    
    print(f"\n   IPS Performance:")
    print(f"     Packets analyzed: {metrics_report['ips']['packets_analyzed']}")
    print(f"     Alerts: {metrics_report['ips']['alerts_generated']}")
    print(f"     Confidence: {metrics_report['ips']['avg_confidence']:.4f}")
    
    # Save metrics report
    with open('results/pipeline_metrics.json', 'w') as f:
        json.dump(metrics_report, f, indent=2)
    print("\n✅ Metrics saved to results/pipeline_metrics.json")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print_section("PIPELINE EXECUTION COMPLETE!")
    
    print("✅ Generated Artifacts:")
    print("\n   📁 Models:")
    print("     - trained_models/ids_model_rf_augmented.pkl")
    print("     - trained_models/ids_model_xgb_augmented.pkl")
    print("     - trained_models/prevention_model.pkl")
    print("     - trained_models/generator_unsw.h5")
    print("     - trained_models/discriminator_unsw.h5")
    
    print("\n   📊 Results:")
    print("     - results/model_comparison_cv.csv")
    print("     - results/pipeline_metrics.json")
    
    print("\n   🔍 Explainability:")
    print("     - results/xai_explanations/shap_values.csv")
    print("     - results/xai_explanations/shap_summary_beeswarm.png")
    print("     - results/xai_explanations/shap_bar_plot.png")
    print("     - results/xai_explanations/shap_force_plot_instance_0.html")
    print("     - results/xai_explanations/lime_explanation_instance_0.html")
    
    print("\n" + "🎉 "*40)
    print("END-TO-END PIPELINE COMPLETED SUCCESSFULLY!")
    print("🎉 "*40 + "\n")
    
    return metrics_report


if __name__ == "__main__":
    try:
        metrics = main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
