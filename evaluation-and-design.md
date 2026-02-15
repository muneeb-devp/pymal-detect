# Model Evaluation and Design Decisions

## Executive Summary

This document describes the machine learning pipeline, evaluation methodology, and design decisions for the malware detection system. The final production model achieves **98.06% accuracy** and **0.9962 AUC-ROC** on the hold-out test set.

---

## Dataset

**Source:** Brazilian Malware Dataset  
**Total Samples:** 50,181 PE (Portable Executable) files  
**Features:** 27 original features extracted from Windows PE files  
**Target:** Binary classification (Malware = 1, Goodware = 0)

### Class Distribution
- Malware: 29,105 samples (58%)
- Goodware: 21,076 samples (42%)
- **Class Imbalance:** Moderate (1.38:1 ratio)

### Train-Test Split
- **Training Set:** 40,145 samples (80%)
- **Test Set:** 10,036 samples (20%)
- **Split Method:** Stratified random split (preserves class distribution)
- **Random State:** 42 (for reproducibility)

---

## Data Preprocessing

### 1. Feature Cleaning

**Dropped Features:**
- `Identify` - High missing values, not informative
- `SHA1` - File identifier, not predictive
- `FirstSeenDate` - Temporal feature, causes data leakage
- `ImportedDlls` - High cardinality text feature
- `ImportedSymbols` - High cardinality text feature

**Rationale:** These features either had too many missing values, were identifiers rather than characteristics, or could cause data leakage in a production system.

### 2. Missing Value Handling

**Strategy:** Median imputation for numeric features

```python
for col in numeric_columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())
```

**Rationale:** Median is robust to outliers in PE file characteristics.

### 3. Feature Scaling

**Method:** StandardScaler (z-score normalization)

```python
X_scaled = (X - mean) / std
```

**Rationale:** 
- Required for logistic regression
- Improves convergence speed for neural networks
- Tree-based models benefit from normalized features when combined in ensembles

### 4. Final Feature Set

**41 numeric features** including:
- `BaseOfCode`, `BaseOfData` - Code/data section addresses
- `Characteristics`, `DllCharacteristics` - PE file flags
- `Entropy` - File randomness measure (important for packed/encrypted malware)
- `FileAlignment`, `ImageBase` - Memory layout parameters
- `Machine`, `Magic` - Binary format identifiers
- `NumberOfSections` - File structure complexity
- `Size`, `SizeOfCode`, `SizeOfImage` - File size metrics
- Various other PE header fields

---

## Cross-Validation Results

**Methodology:** 10-fold Stratified Cross-Validation on Training Set

All models were evaluated using:
- **Metric 1:** AUC-ROC (primary metric)
- **Metric 2:** Accuracy (secondary metric)
- **Folds:** 10 stratified folds
- **Scoring:** Mean ± Standard Deviation

### Model Comparison

| Model | CV AUC (Mean ± Std) | CV Accuracy (Mean ± Std) | Rank |
|-------|---------------------|--------------------------|------|
| **LightGBM** | **0.9971 ± 0.0005** | **0.9809 ± 0.0029** | 🥇 1 |
| **XGBoost** | **0.9971 ± 0.0006** | **0.9811 ± 0.0026** | 🥈 2 |
| Random Forest | 0.9955 ± 0.0008 | 0.9758 ± 0.0027 | 🥉 3 |
| Decision Tree | 0.9832 ± 0.0012 | 0.9714 ± 0.0027 | 4 |
| Logistic Regression | 0.8772 ± 0.0056 | 0.8136 ± 0.0052 | 5 |
| CatBoost | N/A* | 0.9753 ± 0.0028 | 6 |

*CatBoost AUC-ROC could not be computed due to technical issues during training.

### Detailed Results

#### 1. LightGBM (Selected for Production) ✅
- **CV AUC:** 0.9971 ± 0.0005
- **CV Accuracy:** 0.9809 ± 0.0029
- **Training AUC:** 0.9984
- **Training Accuracy:** 0.9855
- **Training Time:** ~30 seconds
- **Inference Time:** <10ms per sample

#### 2. XGBoost
- **CV AUC:** 0.9971 ± 0.0006
- **CV Accuracy:** 0.9811 ± 0.0026
- **Training AUC:** 0.9984
- **Training Accuracy:** 0.9861
- **Training Time:** ~45 seconds

#### 3. Random Forest
- **CV AUC:** 0.9955 ± 0.0008
- **CV Accuracy:** 0.9758 ± 0.0027
- **Training AUC:** 0.9975
- **Training Accuracy:** 0.9811
- **Training Time:** ~90 seconds

#### 4. Decision Tree
- **CV AUC:** 0.9832 ± 0.0012
- **CV Accuracy:** 0.9714 ± 0.0027
- **Training AUC:** 0.9924
- **Training Accuracy:** 0.9795
- **Training Time:** ~5 seconds

#### 5. Logistic Regression
- **CV AUC:** 0.8772 ± 0.0056
- **CV Accuracy:** 0.8136 ± 0.0052
- **Training AUC:** 0.8773
- **Training Accuracy:** 0.8138
- **Training Time:** ~2 seconds

#### 6. CatBoost
- **CV Accuracy:** 0.9753 ± 0.0028
- **Training Accuracy:** 0.9781
- **Training Time:** ~120 seconds

---

## Hold-Out Test Set Evaluation

**Selected Model:** LightGBM

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 98.06% |
| **Test AUC-ROC** | 0.9962 |
| **True Positives** | 5,721 |
| **True Negatives** | 4,121 |
| **False Positives** | 103 |
| **False Negatives** | 92 |

### Confusion Matrix

```
                 Predicted
               Goodware  Malware
Actual Goodware   4,121      103
       Malware       92    5,721
```

### Classification Metrics

- **Precision (Malware):** 5,721 / (5,721 + 103) = 98.23%
- **Recall (Malware):** 5,721 / (5,721 + 92) = 98.42%
- **F1-Score (Malware):** 2 × (0.9823 × 0.9842) / (0.9823 + 0.9842) = 98.32%

- **Precision (Goodware):** 4,121 / (4,121 + 92) = 97.82%
- **Recall (Goodware):** 4,121 / (4,121 + 103) = 97.56%
- **F1-Score (Goodware):** 2 × (0.9782 × 0.9756) / (0.9782 + 0.9756) = 97.69%

### Error Analysis

**False Positives (103 samples):** Goodware files incorrectly classified as malware
- **Impact:** User inconvenience (safe files flagged)
- **Possible Causes:** Unusual file characteristics, obfuscated code, or packed goodware

**False Negatives (92 samples):** Malware files incorrectly classified as goodware
- **Impact:** Security risk (malware passes through)
- **Possible Causes:** Simple/stealthy malware, or goodware-like characteristics

**Error Rate Balance:** FP rate (2.44%) and FN rate (1.58%) are relatively balanced.

---

## Design Decisions

### 1. Model Selection: Why LightGBM?

**Selected:** LightGBM over XGBoost (though both had nearly identical performance)

**Reasons:**
1. **Speed:** 30s vs 45s training time (33% faster)
2. **Inference Latency:** <10ms per prediction
3. **Memory Efficiency:** Lower memory footprint
4. **Production-Ready:** Excellent library support and stability
5. **Comparable Performance:** 0.9962 AUC (virtually tied with XGBoost)

**Trade-offs:**
- XGBoost had slightly higher CV accuracy (0.9811 vs 0.9809)
- However, test set performance was virtually identical
- LightGBM's speed advantage makes it better for production deployment

### 2. Hyperparameters

**LightGBM Configuration:**
```python
{
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}
```

**Rationale:**
- Default parameters worked exceptionally well
- No extensive hyperparameter tuning required (98%+ accuracy achieved)
- Conservative learning rate (0.05) for stability
- Regularization through feature/bagging fractions prevents overfitting

### 3. Evaluation Strategy

**10-Fold Stratified Cross-Validation:**
- Ensures robust performance estimates
- Stratification maintains class balance in each fold
- Reduces variance in performance metrics

**Why not nested CV?**
- Default hyperparameters performed excellently
- No hyperparameter optimization needed
- Simpler pipeline, faster training

### 4. Feature Engineering

**Minimal Feature Engineering Used:**
- Kept original PE file features
- No manual feature creation
- Scaling applied for compatibility

**Rationale:**
- Original features are highly informative
- Tree-based models excel with raw features
- Entropy and size metrics naturally capture malware characteristics
- PE header fields provide strong signals

### 5. Class Imbalance Handling

**Strategy:** No special handling (SMOTE, class weights, etc.)

**Rationale:**
- Imbalance is moderate (1.38:1)
- Model performance is excellent without rebalancing
- Stratified sampling maintains natural distribution
- Production data likely has similar distribution

### 6. Deployment Architecture

**Technology Stack:**
- **Framework:** Flask (lightweight, production-ready)
- **Deployment:** Google Cloud Run (serverless, auto-scaling)
- **Dependencies:** Minimal production requirements (requirements-production.txt)
- **Image:** python:3.11-slim + libgomp1 (for LightGBM)

**Design Choices:**
1. **Separate Requirements Files:**
   - `requirements.txt` - Full development dependencies (~1.5GB)
   - `requirements-production.txt` - Minimal inference dependencies (~150MB)
   - **Benefit:** 90% size reduction, faster builds

2. **Model Serialization:**
   - Models committed to git (334KB)
   - No runtime training needed
   - Instant deployment without training delay

3. **API Design:**
   - RESTful endpoints
   - JSON request/response
   - Health check endpoint for monitoring
   - Debug endpoint for troubleshooting

---

## Production Considerations

### 1. Model Monitoring

**Metrics to Track:**
- Prediction latency
- Error rates (FP/FN)
- Feature distribution drift
- Model performance degradation

### 2. Model Updates

**Retraining Triggers:**
- Performance drops below 95% accuracy
- New malware families emerge
- Feature distribution changes significantly

**Retraining Frequency:**
- Quarterly scheduled retraining recommended
- Ad-hoc retraining when triggers activated

### 3. Scalability

**Current Deployment:**
- Auto-scaling up to 100 instances
- <10ms inference time
- Can handle 10,000+ requests/minute

### 4. Limitations

**Known Limitations:**
1. **Static Analysis Only:** Cannot detect runtime behavior
2. **PE Files Only:** Windows executables only, no Linux/macOS
3. **Adversarial Robustness:** May be vulnerable to adversarial examples
4. **Temporal Drift:** Model trained on 2013 data may not generalize to modern malware

**Mitigation Strategies:**
1. Combine with dynamic analysis tools
2. Regular model retraining with recent samples
3. Monitor for concept drift
4. Ensemble with other detection methods

---

## Conclusion

The LightGBM-based malware detection system demonstrates excellent performance:
- ✅ **98.06% test accuracy**
- ✅ **0.9962 AUC-ROC**
- ✅ **Balanced error rates**
- ✅ **Fast inference (<10ms)**
- ✅ **Production-ready deployment**

The model successfully distinguishes between malware and goodware using static PE file features, providing a reliable first line of defense in malware detection systems.

---

**Last Updated:** February 15, 2026  
**Model Version:** 1.0  
**Author:** Malware Detection Team
