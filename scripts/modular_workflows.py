#!/usr/bin/env python3
"""
Modular Workflow Examples - Use Individual Components

This script shows how to use different components of the system
for specific tasks without running the complete pipeline.

Examples:
1. Feature Selection Only
2. Model Evaluation Only
3. Explainability Only
4. Prevention Only
5. IPS Monitoring Only
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import DataPreprocessor
from src.models import ModelTrainer
from src.evaluation import stratified_kfold_evaluation, compare_models_with_cv
from src.xai import SHAPExplainer, generate_shap_csv, LIMEExplainer
from src.prevention import PreventionModel
from src.core import PredictionEngine, RealTimeIPS
from src.features import FeatureExtractor
from src.utils import ensure_directory, MetricsLogger


def print_workflow(title):
    """Print workflow header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


# =============================================================================
# WORKFLOW 1: FEATURE SELECTION & ANALYSIS ONLY
# =============================================================================
def workflow_feature_selection():
    """Select and analyze important features."""
    print_workflow("WORKFLOW 1: Feature Selection & Analysis Only")
    
    print("📊 Loading preprocessed data...")
    preprocessor = DataPreprocessor()
    X_train, y_train, X_test, y_test = preprocessor.process_pipeline()
    
    print(f"✅ Loaded data: {X_train.shape}")
    
    # Train quick model for feature analysis
    print("\n🧠 Training quick model for SHAP analysis...")
    trainer = ModelTrainer()
    model = trainer.train_random_forest(X_train, y_train, n_estimators=50, verbose=False)
    
    # Generate feature importance
    print("🔍 Generating SHAP-based feature importance...")
    ensure_directory('results/xai_explanations')
    
    importance_df = generate_shap_csv(model, X_test)
    
    print("\n📌 Top 15 Important Features:")
    print(importance_df.head(15).to_string(index=False))
    
    # Save results
    importance_df.to_csv('results/feature_importance.csv', index=False)
    print("\n✅ Feature importance saved to results/feature_importance.csv")
    
    return importance_df


# =============================================================================
# WORKFLOW 2: MODEL EVALUATION ONLY
# =============================================================================
def workflow_model_evaluation():
    """Evaluate pretrained models."""
    print_workflow("WORKFLOW 2: Model Evaluation Only")
    
    print("📊 Loading data...")
    preprocessor = DataPreprocessor()
    X_train, y_train, X_test, y_test = preprocessor.process_pipeline()
    
    print(f"✅ Loaded data: Train {X_train.shape}, Test {X_test.shape}")
    
    # Evaluate models
    print("\n📈 Evaluating models with stratified 5-fold CV...")
    
    print("\n1️⃣  Random Forest Evaluation:")
    rf_results = stratified_kfold_evaluation(
        X_train, y_train, X_test, y_test,
        model_type='rf', n_splits=5
    )
    
    print(f"   Test Accuracy: {rf_results['test_metrics']['accuracy']:.4f}")
    print(f"   Test Recall: {rf_results['test_metrics']['recall']:.4f}")
    print(f"   Test F1: {rf_results['test_metrics']['f1_weighted']:.4f}")
    
    print("\n2️⃣  XGBoost Evaluation:")
    xgb_results = stratified_kfold_evaluation(
        X_train, y_train, X_test, y_test,
        model_type='xgb', n_splits=5
    )
    
    print(f"   Test Accuracy: {xgb_results['test_metrics']['accuracy']:.4f}")
    print(f"   Test Recall: {xgb_results['test_metrics']['recall']:.4f}")
    print(f"   Test F1: {xgb_results['test_metrics']['f1_weighted']:.4f}")
    
    # Compare models
    print("\n📊 Model Comparison:")
    comparison_df = compare_models_with_cv(X_train, y_train, X_test, y_test)
    print(comparison_df.to_string())
    
    comparison_df.to_csv('results/model_evaluation_comparison.csv', index=False)
    print("\n✅ Comparison saved to results/model_evaluation_comparison.csv")


# =============================================================================
# WORKFLOW 3: EXPLAINABILITY ANALYSIS ONLY
# =============================================================================
def workflow_explainability():
    """Analyze model predictions with SHAP and LIME."""
    print_workflow("WORKFLOW 3: Explainability Analysis Only")
    
    print("📊 Loading data and pretrained model...")
    preprocessor = DataPreprocessor()
    X_train, y_train, X_test, y_test = preprocessor.process_pipeline()
    
    # Load model
    model_path = 'trained_models/ids_model_rf_augmented.pkl'
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found at {model_path}")
        print("   Training new model...")
        trainer = ModelTrainer()
        model = trainer.train_random_forest(X_train, y_train, n_estimators=100, verbose=False)
    else:
        from src.utils import load_model
        model = load_model(model_path)
    
    print("✅ Model loaded")
    
    # SHAP Analysis
    print("\n🔍 SHAP Explainability Analysis:")
    ensure_directory('results/xai_explanations')
    
    print("   Generating SHAP values...")
    shap_explainer = SHAPExplainer(model, X_test[:100], model_type='tree')
    feature_importance = shap_explainer.get_feature_importance()
    
    print("\n   Top 10 Features by SHAP:")
    print(feature_importance.head(10).to_string(index=False))
    
    print("\n   Saving SHAP plots...")
    shap_explainer.save_all_plots('results/xai_explanations')
    print("   ✅ SHAP plots saved")
    
    # LIME Analysis
    print("\n🔍 LIME Explainability Analysis:")
    
    lime_explainer = LIMEExplainer(model, X_train, feature_names=X_test.columns.tolist())
    
    print("   Explaining predictions for sample instances...")
    for i in range(min(3, len(X_test))):
        print(f"   - Sample {i}...")
        lime_explainer.save_explanation_html(
            X_test.iloc[i],
            f'results/xai_explanations/lime_explanation_instance_{i}.html'
        )
    
    print("✅ LIME explanations saved")


# =============================================================================
# WORKFLOW 4: PREVENTION MODEL TRAINING ONLY
# =============================================================================
def workflow_prevention_training():
    """Train prevention models."""
    print_workflow("WORKFLOW 4: Prevention Model Training Only")
    
    print("📊 Loading augmented data...")
    preprocessor = DataPreprocessor()
    X_train, y_train, X_test, y_test = preprocessor.process_pipeline()
    
    print(f"✅ Loaded data: {X_train.shape}")
    
    # Get top features from SHAP
    print("\n🔍 Getting top features...")
    importance_df = pd.read_csv('results/feature_importance.csv')
    top_features = importance_df['feature'].head(10).tolist()
    print(f"   Using features: {top_features}")
    
    # Train prevention model
    print("\n🛡️  Training prevention model...")
    prevention_model = PreventionModel(model_type='xgboost')
    prevention_model.train(
        X_train, y_train,
        top_features=top_features,
        n_estimators=100,
        max_depth=5
    )
    
    # Evaluate
    print("\n📊 Evaluating prevention model...")
    y_pred = prevention_model.predict(X_test[top_features])
    
    accuracy = (y_pred == y_test.values).mean()
    print(f"   Test Accuracy: {accuracy:.4f}")
    
    # Save model
    ensure_directory('trained_models')
    prevention_model.save('trained_models/prevention_model_trained.pkl')
    print("\n✅ Prevention model saved")


# =============================================================================
# WORKFLOW 5: REAL-TIME IPS MONITORING ONLY
# =============================================================================
def workflow_real_time_ips():
    """Monitor traffic in real-time with IPS."""
    print_workflow("WORKFLOW 5: Real-Time IPS Monitoring Only")
    
    print("🔌 Creating IPS engine...")
    
    # Load prediction engine
    engine = PredictionEngine(
        'trained_models/ids_model_rf_augmented.pkl',
        'data/processed/unsw_scaler.pkl'
    )
    
    # Create IPS
    ips = RealTimeIPS(engine, alert_threshold=0.7)
    print("✅ IPS engine created")
    
    # Simulate traffic
    print("\n🚦 Simulating real-time traffic monitoring...")
    
    preprocessor = DataPreprocessor()
    X_train, y_train, X_test, y_test = preprocessor.process_pipeline()
    
    print(f"   Monitoring {min(50, len(X_test))} packets...\n")
    
    alerts = []
    for idx in range(min(50, len(X_test))):
        packet = X_test.iloc[idx].values
        is_attack, alert_info = ips.analyze_packet(packet)
        
        if alert_info:
            alerts.append({
                'packet_id': idx,
                'confidence': alert_info.get('confidence'),
                'true_label': int(y_test.iloc[idx]),
                'detected_as_attack': is_attack
            })
            
            if len(alerts) <= 5:  # Show first 5 alerts
                print(f"   🚨 Alert {len(alerts)}: Packet {idx}")
                print(f"      Confidence: {alert_info.get('confidence'):.4f}")
                print(f"      Actual: {'Attack' if y_test.iloc[idx] == 1 else 'Normal'}")
    
    # Get summary
    summary = ips.get_alert_summary()
    
    print(f"\n📊 IPS Summary:")
    print(f"   Packets analyzed: {summary['total_packets']}")
    print(f"   Alerts generated: {summary['total_alerts']}")
    print(f"   Alert rate: {summary['total_alerts']/summary['total_packets']*100:.2f}%")
    print(f"   Avg confidence: {summary['avg_confidence']:.4f}")
    
    # Save alerts
    if alerts:
        alerts_df = pd.DataFrame(alerts)
        alerts_df.to_csv('results/ips_alerts.csv', index=False)
        print("\n✅ Alerts saved to results/ips_alerts.csv")


# =============================================================================
# WORKFLOW 6: FEATURE EXTRACTION FROM PACKETS
# =============================================================================
def workflow_feature_extraction():
    """Extract features from raw network packets."""
    print_workflow("WORKFLOW 6: Feature Extraction from Packets")
    
    print("🔌 Creating feature extractor...")
    extractor = FeatureExtractor()
    
    # Simulate raw packets
    print("\n📦 Simulating raw network packets...\n")
    
    packets = [
        {
            'src_ip': '192.168.1.100',
            'dst_ip': '8.8.8.8',
            'src_port': 54321,
            'dst_port': 443,
            'proto': 6,
            'ttl': 64,
            'payload_size': 1024,
            'flags': 'S'
        },
        {
            'src_ip': '192.168.1.200',
            'dst_ip': '1.1.1.1',
            'src_port': 53248,
            'dst_port': 80,
            'proto': 6,
            'ttl': 128,
            'payload_size': 512,
            'flags': 'SA'
        },
        {
            'src_ip': '10.0.0.50',
            'dst_ip': '192.168.1.1',
            'src_port': 12345,
            'dst_port': 22,
            'proto': 6,
            'ttl': 255,
            'payload_size': 256,
            'flags': 'F'
        }
    ]
    
    print("Extracted Features from Packets:")
    print("-" * 80)
    
    for i, packet in enumerate(packets, 1):
        features = extractor.extract_from_packet(packet)
        
        print(f"\nPacket {i}:")
        print(f"  Source: {packet['src_ip']}:{packet['src_port']} → ", end="")
        print(f"{packet['dst_ip']}:{packet['dst_port']}")
        print(f"  Protocol: {features['proto']}")
        print(f"  Service: {features['service']}")
        print(f"  State: {features['state']}")
        print(f"  Size: {features['sbytes']} bytes")
        print(f"  TTL: {features['sttl']}")
    
    # Batch extraction
    print("\n\n📊 Batch Feature Extraction:")
    print(f"   Total packets: {len(packets)}")
    
    features_df = extractor.extract_batch(packets)
    print(f"   Extracted features: {features_df.shape}")
    print(f"   Features: {list(features_df.columns[:5])}... (showing first 5)")
    
    features_df.to_csv('results/extracted_features.csv', index=False)
    print("\n✅ Extracted features saved to results/extracted_features.csv")


# =============================================================================
# MAIN MENU
# =============================================================================
def main():
    """Run modular workflows."""
    
    print("\n" + "="*80)
    print("  MODULAR WORKFLOW EXAMPLES")
    print("="*80)
    
    workflows = {
        '1': ('Feature Selection & Analysis', workflow_feature_selection),
        '2': ('Model Evaluation', workflow_model_evaluation),
        '3': ('Explainability Analysis (SHAP + LIME)', workflow_explainability),
        '4': ('Prevention Model Training', workflow_prevention_training),
        '5': ('Real-Time IPS Monitoring', workflow_real_time_ips),
        '6': ('Feature Extraction from Packets', workflow_feature_extraction),
        '0': ('Run All Workflows', None)
    }
    
    print("\nAvailable Workflows:")
    for key, (name, _) in workflows.items():
        print(f"  {key}. {name}")
    
    print("\nUsage:")
    print("  python scripts/modular_workflows.py <workflow_number>")
    print("  python scripts/modular_workflows.py 0  # Run all")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nSelect workflow (0-6): ").strip()
    
    ensure_directory('results')
    
    if choice == '0':
        # Run all
        print("\n🚀 Running all workflows...\n")
        workflow_feature_selection()
        workflow_model_evaluation()
        workflow_explainability()
        workflow_prevention_training()
        workflow_real_time_ips()
        workflow_feature_extraction()
    elif choice in workflows and workflows[choice][1]:
        # Run selected
        try:
            workflows[choice][1]()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ Invalid choice")
        sys.exit(1)
    
    print("\n✅ Workflow(s) completed!")


if __name__ == "__main__":
    main()
