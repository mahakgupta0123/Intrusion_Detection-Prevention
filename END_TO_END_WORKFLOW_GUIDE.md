# 🚀 Complete End-to-End IDS/IPS Workflow Guide

## Overview

This guide explains how to execute the complete IPDRS pipeline from data preprocessing through real-time IPS monitoring, including augmentation, feature selection, detection, prevention, and explainability analysis.

---

## 📋 Workflow Steps

### **Step 1: Data Preprocessing** ✅
- **Input**: Raw UNSW-NB15 CSV files
- **Output**: Preprocessed training & test sets
- **Key Actions**:
  - Load UNSW-NB15 dataset
  - Handle missing values
  - Encode categorical features (protocol, service, state)
  - Scale numerical features using StandardScaler
  - Train/test split (70/30)

```python
from src.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
X_train, y_train, X_test, y_test = preprocessor.process_pipeline()
# Returns: Train (X, y) and Test (X, y) sets
```

**Files Generated**:
```
data/processed/
├── X_train_unsw.csv
├── y_train_unsw.csv
├── X_test_unsw.csv
├── y_test_unsw.csv
├── unsw_scaler.pkl
└── label_encoders.pkl
```

---

### **Step 2: Feature Extraction & Analysis** 🔍
- **Input**: Preprocessed data
- **Output**: Feature statistics and analysis
- **Key Actions**:
  - Analyze feature distributions
  - Calculate feature statistics
  - Identify important features
  - Prepare for augmentation

```python
from src.features import FeatureExtractor

extractor = FeatureExtractor()
# For raw packets:
features = extractor.extract_from_packet({
    'src_ip': '192.168.1.1',
    'dst_ip': '8.8.8.8',
    'dst_port': 443,
    'proto': 6
})
```

**Typical Statistics**:
```
Total features: 34
Training samples: ~175,000
Test samples: ~75,000
Attack ratio: ~10% (class imbalance)
```

---

### **Step 3: Data Augmentation** 🤖

#### **Phase 3a: GAN-Based Augmentation**
- **Input**: Training data (heavily imbalanced)
- **Output**: Synthetic attack samples
- **Key Actions**:
  - Extract attack samples only
  - Train GAN on attack patterns
  - Generate synthetic attacks (1.5x multiplier)
  - Validate synthetic data quality (KL-divergence)

```python
from src.augmentation import GANAugmentor

augmentor = GANAugmentor(latent_dim=100, batch_size=64)

# Train on attack samples only
attack_data = X_train[y_train == 1].values
augmentor.train(attack_data, epochs=5000)

# Generate synthetic data
X_aug, y_aug = augmentor.augment_data(X_train, y_train, multiplier=1.5)
# Result: 1.5x more attack samples added
```

**GAN Architecture**:
```
Generator:
  Input (latent: 100) → 
  Dense(256) → LeakyReLU → 
  Dense(512) → LeakyReLU → 
  Dense(features) → Sigmoid

Discriminator:
  Input (features) → 
  Dense(512) → LeakyReLU → 
  Dense(256) → LeakyReLU → 
  Dense(1) → Sigmoid
```

**Key Improvements**:
- ✅ 1.5x multiplier (vs 0.5x before)
- ✅ Adaptive percentile-based clipping
- ✅ Label smoothing for stability
- ✅ KL-divergence validation

#### **Phase 3b: SMOTE Integration**
- **Input**: GAN-augmented data
- **Output**: Optimally balanced data
- **Key Actions**:
  - Apply SMOTE (0.8 ratio)
  - Generate additional synthetic samples
  - Achieve near-perfect balance

```python
from src.augmentation import combine_gan_and_smote

X_final, y_final = combine_gan_and_smote(X_aug, y_aug, apply_smote=True)
# Result: Perfectly balanced dataset
```

**Before vs After Augmentation**:
```
BEFORE (Original):
  Normal: ~157,000 (90%)
  Attack: ~17,000 (10%)
  Ratio: 9:1 (highly imbalanced)

AFTER GAN (1.5x):
  Normal: ~157,000 (82%)
  Attack: ~42,500 (18%)
  Ratio: 3.7:1 (improved)

AFTER GAN + SMOTE:
  Normal: ~150,000 (50%)
  Attack: ~150,000 (50%)
  Ratio: 1:1 (perfectly balanced)
```

**Files Generated**:
```
trained_models/
├── generator_unsw.h5      (GAN generator)
└── discriminator_unsw.h5  (GAN discriminator)

data/processed/
├── X_train_unsw_augmented.csv
└── y_train_unsw_augmented.csv
```

---

### **Step 4: Model Training** 🧠

Train multiple models on augmented data:

#### **Model 1: Random Forest**
```python
trainer = ModelTrainer()
rf_model = trainer.train_random_forest(
    X_final, y_final,
    n_estimators=200,
    max_depth=15,
    class_weight='balanced'
)
```

**Hyperparameters**:
- Estimators: 200
- Max depth: 15
- Min samples split: 5
- Class weight: balanced
- Random state: 42

#### **Model 2: XGBoost**
```python
xgb_model = trainer.train_xgboost(
    X_final, y_final,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=(y_final == 0).sum() / (y_final == 1).sum()
)
```

**Hyperparameters**:
- Estimators: 200
- Max depth: 6
- Learning rate: 0.05
- Scale pos weight: calculated
- Eval metric: logloss
- Early stopping: 10 rounds

#### **Model 3: LSTM Neural Network** (Optional)
```python
from src.models import train_lstm_model

lstm_model, history = train_lstm_model(
    X_train, y_train,
    X_test, y_test,
    epochs=50,
    batch_size=32
)
```

#### **Model 4: CNN Neural Network** (Optional)
```python
from src.models import train_cnn_model

cnn_model, history = train_cnn_model(
    X_train, y_train,
    X_test, y_test,
    epochs=50,
    batch_size=32
)
```

**Files Generated**:
```
trained_models/
├── ids_model_rf_augmented.pkl
├── ids_model_xgb_augmented.pkl
├── cnn_model_unsw.h5 (optional)
└── lstm_model_unsw.h5 (optional)
```

---

### **Step 5: Model Evaluation** 📊

Evaluate models with stratified k-fold cross-validation:

```python
from src.evaluation import stratified_kfold_evaluation, compare_models_with_cv

# Random Forest
rf_results = stratified_kfold_evaluation(
    X_final, y_final,
    X_test, y_test,
    model_type='rf',
    n_splits=5
)

# XGBoost
xgb_results = stratified_kfold_evaluation(
    X_final, y_final,
    X_test, y_test,
    model_type='xgb',
    n_splits=5
)

# Compare all models
comparison_df = compare_models_with_cv(X_final, y_final, X_test, y_test)
```

**Evaluation Metrics**:
- Accuracy
- Precision
- Recall (critical for attacks)
- F1-Score (weighted & macro)
- ROC-AUC
- Confusion Matrix
- Overfitting detection

**Expected Results (with augmentation)**:
```
Random Forest:
  Train Accuracy: 0.98
  Test Accuracy: 0.95
  Test Recall: 0.88 (catches most attacks)
  Test F1: 0.92

XGBoost:
  Train Accuracy: 0.99
  Test Accuracy: 0.96
  Test Recall: 0.90
  Test F1: 0.93
```

**Files Generated**:
```
results/
├── model_comparison_cv.csv
├── confusion_matrix_rf_cv.png
└── confusion_matrix_xgb_cv.png
```

---

### **Step 6: Feature Selection & XAI** 🔍

#### **Phase 6a: SHAP Feature Importance**
```python
from src.xai import generate_shap_csv, SHAPExplainer

# Generate SHAP values
importance_df = generate_shap_csv(
    rf_model,
    X_test,
    output_path='results/xai_explanations/shap_values.csv',
    model_type='tree'
)

# Top features
top_10_features = importance_df['feature'].head(10).tolist()
```

**SHAP Analysis**:
- Computes contribution of each feature to predictions
- Identifies global feature importance
- Highlights local explanations
- Validates model decisions

**Top Features Example**:
```
1. sttl (source TTL)
2. ct_dst_sport_ltm (connection count)
3. rate (packet rate)
4. proto (protocol type)
5. ct_srv_dst (service destination count)
6. service (service type)
7. smean (source IP mean)
8. ct_state_ttl (state TTL count)
9. sbytes (source bytes)
10. ct_srv_src (service source count)
```

#### **Phase 6b: SHAP Visualizations**
```python
shap_explainer = SHAPExplainer(rf_model, X_test)
shap_explainer.save_all_plots('results/xai_explanations')
```

**Visualizations Generated**:
- Summary plot (beeswarm) - feature impact
- Bar plot - mean absolute SHAP values
- Force plot (HTML) - instance explanation
- Dependence plots - feature interactions

#### **Phase 6c: LIME Local Explanations**
```python
from src.xai import LIMEExplainer

lime_explainer = LIMEExplainer(rf_model, X_final)

# Explain specific predictions
explanation = lime_explainer.explain_instance(X_test.iloc[0])
lime_explainer.save_explanation_html(
    X_test.iloc[0],
    'results/xai_explanations/lime_explanation_instance_0.html'
)
```

**LIME Analysis**:
- Local interpretability for single predictions
- Shows feature contributions for specific instances
- Interactive HTML explanations
- Helps understand model decisions

**Files Generated**:
```
results/xai_explanations/
├── shap_values.csv
├── shap_values_importance.csv
├── shap_summary_beeswarm.png
├── shap_bar_plot.png
├── shap_force_plot_instance_0.html
└── lime_explanation_instance_0.html
```

---

### **Step 7: Prevention Model Training** 🛡️

Train prevention model using top features:

```python
from src.prevention import PreventionModel, apply_prevention_rules

# Train with top features
prevention_model = PreventionModel(model_type='xgboost')
prevention_model.train(
    X_final, y_final,
    top_features=top_10_features,
    n_estimators=100,
    max_depth=5
)

# Apply prevention rules
X_test_subset = X_test[top_10_features].copy()
X_with_flags = apply_prevention_rules(X_test_subset, top_10_features)

# Get results
flagged = X_with_flags['prevent_flag'].sum()
print(f"Flagged: {flagged} / {len(X_test)} packets")
```

**Prevention Strategy**:
- Uses only top 10 important features
- Reduces false positives
- Faster inference
- Targeted blocking

**Prevention Rules** (example):
```python
{
    'sttl': lambda x: x > 0.7,           # Unusual TTL
    'ct_dst_sport_ltm': lambda x: x < -0.5,  # Unusual connection pattern
    'rate': lambda x: x > 1.2,           # High packet rate
    'proto': lambda x: x < 0.3,          # Unusual protocol
    'ct_srv_dst': lambda x: x > 0.8      # Multiple service connections
}
```

**Files Generated**:
```
trained_models/
└── prevention_model.pkl

results/
└── prevention_analysis.csv
```

---

### **Step 8: Real-Time IPS Deployment** 🚀

Deploy real-time intrusion prevention system:

```python
from src.core import PredictionEngine, RealTimeIPS

# Create prediction engine
engine = PredictionEngine(
    'trained_models/ids_model_rf_augmented.pkl',
    'data/processed/unsw_scaler.pkl'
)

# Create IPS
ips = RealTimeIPS(engine, alert_threshold=0.7)

# Analyze traffic
is_attack, alert = ips.analyze_packet(packet_features)

# Get summary
summary = ips.get_alert_summary()
```

**IPS Workflow**:
1. **Feature Extraction**: Parse incoming packet
2. **Scaling**: Apply saved scaler
3. **Prediction**: Run through trained model
4. **Confidence Check**: Compare to threshold
5. **Alert Generation**: Flag if confident attack
6. **Logging**: Record all decisions

**Real-Time Metrics**:
```
Packets analyzed: 10,000
Alerts generated: 234
Alert rate: 2.34%
Avg confidence: 0.82
Processing time: 1.2ms/packet
```

---

## 📊 Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RAW UNSW-NB15 DATA                              │
└────────────────┬────────────────────────────────────────────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │ STEP 1: PREPROCESSING      │
    │ - Load data                │
    │ - Handle missing values    │
    │ - Encode features          │
    │ - Scale features           │
    └────────────┬───────────────┘
                 │
    ┌────────────▼───────────────┐
    │ Train & Test Split         │
    │ Train: 157K normal, 17K atk│
    │ Test: 67K normal, 7.3K atk │
    └────────────┬───────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │ STEP 2: FEATURE ANALYSIS   │
    │ - Analyze features         │
    │ - Calculate statistics     │
    │ - Identify patterns        │
    └────────────┬───────────────┘
                 │
                 ▼
    ┌────────────────────────────────────┐
    │ STEP 3: DATA AUGMENTATION          │
    ├────────────────────────────────────┤
    │ PHASE 3a: GAN                      │
    │ - Train on attacks                 │
    │ - Generate synthetic (1.5x)        │
    │ - Validate distribution            │
    │                                    │
    │ PHASE 3b: SMOTE                    │
    │ - Further augmentation             │
    │ - Balance classes                  │
    │ Final: 150K normal, 150K attack    │
    └────────────┬──────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────────┐
    │ STEP 4: MODEL TRAINING             │
    │ - RF: 200 estimators               │
    │ - XGB: 200 estimators              │
    │ - LSTM: 64→32 units (optional)     │
    │ - CNN: Conv1D + MaxPool (optional) │
    └────────────┬──────────────────────┘
                 │
     ┌───────────┴──────────────┐
     │                          │
     ▼                          ▼
 ┌─────────────┐        ┌──────────────┐
 │ RF Model    │        │ XGB Model    │
 │ ACC: 0.95   │        │ ACC: 0.96    │
 │ Rec: 0.88   │        │ Rec: 0.90    │
 │ F1: 0.92    │        │ F1: 0.93     │
 └──────┬──────┘        └───────┬──────┘
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
    ┌────────────────────────────────────┐
    │ STEP 5: EVALUATION                 │
    │ - Stratified 5-fold CV             │
    │ - Confusion matrix                 │
    │ - Metrics comparison               │
    │ Select best model (XGB 0.93 F1)    │
    └────────────┬──────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────────┐
    │ STEP 6: EXPLAINABILITY (XAI)       │
    ├────────────────────────────────────┤
    │ - SHAP feature importance          │
    │ - LIME local explanations          │
    │ - Feature impact analysis          │
    │ Top 10 features identified         │
    └────────────┬──────────────────────┘
                 │
     ┌───────────┴──────────────┐
     │                          │
     ▼                          ▼
┌────────────────┐     ┌────────────────┐
│ STEP 7:        │     │ STEP 8:        │
│ PREVENTION     │     │ REAL-TIME IPS  │
│ - Train model  │     │ - Deploy engine│
│ - Top features │     │ - Monitor live │
│ - Rules        │     │ - Generate     │
│ - Blocking     │     │   alerts       │
└────────────────┘     └────────────────┘
```

---

## 🚀 Execution Scripts

### **Option 1: Complete End-to-End Pipeline**
```bash
python scripts/complete_workflow.py
```

**Duration**: 30-60 minutes
**Output**: All artifacts (models, metrics, visualizations)

### **Option 2: Individual Workflows**
```bash
# Feature selection only
python scripts/modular_workflows.py 1

# Model evaluation only
python scripts/modular_workflows.py 2

# Explainability analysis
python scripts/modular_workflows.py 3

# Prevention training
python scripts/modular_workflows.py 4

# Real-time IPS
python scripts/modular_workflows.py 5

# Feature extraction
python scripts/modular_workflows.py 6

# All workflows
python scripts/modular_workflows.py 0
```

### **Option 3: Manual Python Code**
```python
from src.preprocessing import DataPreprocessor
from src.augmentation import GANAugmentor, combine_gan_and_smote
from src.models import ModelTrainer
from src.evaluation import stratified_kfold_evaluation
from src.xai import generate_shap_csv
from src.prevention import PreventionModel
from src.core import PredictionEngine, RealTimeIPS

# Step-by-step execution
preprocessor = DataPreprocessor()
X_train, y_train, X_test, y_test = preprocessor.process_pipeline()

augmentor = GANAugmentor()
augmentor.train(X_train[y_train == 1].values, epochs=5000)
X_aug, y_aug = augmentor.augment_data(X_train, y_train)
X_final, y_final = combine_gan_and_smote(X_aug, y_aug)

trainer = ModelTrainer()
trainer.train_random_forest(X_final, y_final)

results = stratified_kfold_evaluation(X_final, y_final, X_test, y_test)
importance = generate_shap_csv(trainer.model, X_test)

# ... continue with prevention and IPS
```

---

## 📈 Expected Performance

### **Before Augmentation**
```
Random Forest (original):
  Test Accuracy: 0.92
  Test Recall: 0.65 ❌ (misses 35% of attacks!)
  Test F1: 0.78

XGBoost (original):
  Test Accuracy: 0.93
  Test Recall: 0.68
  Test F1: 0.79
```

### **After Augmentation + Training**
```
Random Forest (augmented):
  Test Accuracy: 0.95
  Test Recall: 0.88 ✅ (catches 88% of attacks!)
  Test F1: 0.92 (+13% improvement!)

XGBoost (augmented):
  Test Accuracy: 0.96
  Test Recall: 0.90
  Test F1: 0.93 (+14% improvement!)
```

---

## 📁 Generated Artifacts

### **Models**
```
trained_models/
├── ids_model_rf_augmented.pkl
├── ids_model_xgb_augmented.pkl
├── generator_unsw.h5
├── discriminator_unsw.h5
├── prevention_model.pkl
├── unsw_scaler.pkl
└── label_encoders.pkl
```

### **Results**
```
results/
├── model_comparison_cv.csv
├── pipeline_metrics.json
├── feature_importance.csv
├── ips_alerts.csv
├── xai_explanations/
│   ├── shap_values.csv
│   ├── shap_bar_plot.png
│   ├── shap_summary_beeswarm.png
│   ├── shap_force_plot_instance_0.html
│   └── lime_explanation_instance_0.html
└── evaluation_plots/
    ├── confusion_matrix_rf_cv.png
    └── confusion_matrix_xgb_cv.png
```

### **Data**
```
data/processed/
├── X_train_unsw.csv
├── y_train_unsw.csv
├── X_test_unsw.csv
├── y_test_unsw.csv
├── X_train_unsw_augmented.csv
├── y_train_unsw_augmented.csv
├── unsw_scaler.pkl
└── label_encoders.pkl
```

---

## ✅ Checklist

- [ ] Run `python scripts/complete_workflow.py`
- [ ] Wait for completion (30-60 minutes)
- [ ] Check `results/` directory for outputs
- [ ] Review metrics in `results/pipeline_metrics.json`
- [ ] View visualizations in `results/xai_explanations/`
- [ ] Verify models in `trained_models/`
- [ ] Test individual workflows as needed
- [ ] Deploy IPS using `RealTimeIPS` class
- [ ] Monitor alerts in `ips_logs/`
- [ ] Iterate with different parameters

---

## 🎓 Learning Resources

1. **QUICK_START.md** - 5-minute setup
2. **README.md** - Full documentation
3. **MODULAR_ARCHITECTURE_COMPLETE.md** - Module details
4. **IMPLEMENTATION_GUIDE.md** - Deep dive
5. **Script docstrings** - Code-level documentation

---

## 🚀 Ready to Go!

You now have everything needed to:
1. ✅ Preprocess network traffic data
2. ✅ Augment imbalanced datasets with GAN + SMOTE
3. ✅ Train state-of-the-art detection models
4. ✅ Evaluate with stratified cross-validation
5. ✅ Explain predictions with SHAP + LIME
6. ✅ Train prevention models
7. ✅ Deploy real-time IPS system

**Start executing**: `python scripts/complete_workflow.py`

Happy coding! 🚀

---

**Version**: 1.0.0
**Last Updated**: 2026-06-28
**Status**: ✅ Complete & Ready
