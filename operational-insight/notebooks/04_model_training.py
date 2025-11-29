# ============================================================================
# MILESTONE 3: MODEL TRAINING
# ============================================================================
# Goal: Train Decision Tree and Random Forest WITHOUT proxy features
# Strategy: Use only features that work on both 5k and 2k naturally
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MILESTONE 3: MODEL TRAINING")
print("="*80)

# ============================================================================
# LOAD PROCESSED 5K DATASET
# ============================================================================

df_5k = pd.read_csv('operational-insight/data/processed/5k_with_proxies.csv')
print(f"\n✓ Loaded 5k dataset: {df_5k.shape}")

# ============================================================================
# DEFINE FEATURE SET (NO PROXIES!)
# ============================================================================

print("\n" + "="*80)
print("FEATURE SET SELECTION")
print("="*80)

# Based on Milestone 1 EDA and Milestone 2 findings:
# Use ONLY features that don't need proxies and have signal

features_to_use = [
    'variance_to_expected',     # 10.02x lift - CRITICAL ⭐⭐⭐
    'duration_minutes',          # 2.07x lift - IMPORTANT ⭐
    'wait_time_minutes',         # 1.65x lift - IMPORTANT ⭐
    'handoff_count_so_far',      # 0.97x lift - WEAK but available
    'hour_of_day',               # Temporal pattern
    'sla_breached',              # Business context
]

# We're EXCLUDING:
# - queue_length_at_start (needs proxy, proxy failed)
# - system_load_index_0to1 (needs proxy, proxy failed)
# - weekday (one-hot encoded would create too many features for 50 samples)
# - priority (would need encoding, adds complexity)

print("\nFeatures for modeling:")
for i, feat in enumerate(features_to_use, 1):
    if feat in df_5k.columns:
        print(f"  {i}. ✓ {feat}")
    else:
        print(f"  {i}. ✗ {feat} - MISSING!")

print(f"\nTotal features: {len(features_to_use)}")

# ============================================================================
# PREPARE DATA
# ============================================================================

print("\n" + "="*80)
print("PREPARE TRAINING DATA")
print("="*80)

# Check for missing values in features
missing_check = df_5k[features_to_use].isnull().sum()
print("\nMissing values in features:")
print(missing_check[missing_check > 0] if missing_check.sum() > 0 else "None")

# Create feature matrix and target
X = df_5k[features_to_use].copy()
y = df_5k['is_bottleneck_event'].copy()

# Handle any missing values (fill with median)
X = X.fillna(X.median())

print(f"\n✓ Feature matrix X: {X.shape}")
print(f"✓ Target vector y: {y.shape}")
print(f"\nClass distribution:")
print(f"  Normal (0): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
print(f"  Bottleneck (1): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")

# ============================================================================
# MODEL 1: DECISION TREE
# ============================================================================

print("\n" + "="*80)
print("MODEL 1: DECISION TREE")
print("="*80)

# Simple Decision Tree (not too complex given small data)
dt_model = DecisionTreeClassifier(
    max_depth=4,              # Shallow to prevent overfitting
    min_samples_leaf=3,       # Need at least 3 samples per leaf
    class_weight='balanced',  # Handle class imbalance
    random_state=42
)

# Cross-validation
print("\nPerforming 5-fold stratified cross-validation...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results_dt = cross_validate(
    dt_model, X, y, cv=cv,
    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    return_train_score=True
)

print("\n--- Decision Tree Cross-Validation Results ---")
print(f"Accuracy:  {cv_results_dt['test_accuracy'].mean():.3f} ± {cv_results_dt['test_accuracy'].std():.3f}")
print(f"Precision: {cv_results_dt['test_precision'].mean():.3f} ± {cv_results_dt['test_precision'].std():.3f}")
print(f"Recall:    {cv_results_dt['test_recall'].mean():.3f} ± {cv_results_dt['test_recall'].std():.3f}")
print(f"F1:        {cv_results_dt['test_f1'].mean():.3f} ± {cv_results_dt['test_f1'].std():.3f}")
print(f"ROC-AUC:   {cv_results_dt['test_roc_auc'].mean():.3f} ± {cv_results_dt['test_roc_auc'].std():.3f}")

# Check for overfitting
train_acc = cv_results_dt['train_accuracy'].mean()
test_acc = cv_results_dt['test_accuracy'].mean()
overfit_gap = train_acc - test_acc

print(f"\nOverfitting check:")
print(f"  Train accuracy: {train_acc:.3f}")
print(f"  Test accuracy:  {test_acc:.3f}")
print(f"  Gap:            {overfit_gap:.3f}")

if overfit_gap > 0.15:
    print("  ⚠️  Model may be overfitting (gap > 0.15)")
elif overfit_gap > 0.10:
    print("  ⚠️  Moderate overfitting (0.10 < gap < 0.15)")
else:
    print("  ✓ Acceptable generalization (gap < 0.10)")

# Train on full data for final model
dt_model.fit(X, y)

# Feature importances
dt_importances = pd.DataFrame({
    'feature': features_to_use,
    'importance': dt_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n--- Decision Tree Feature Importances ---")
print(dt_importances.to_string(index=False))

# ============================================================================
# MODEL 2: RANDOM FOREST
# ============================================================================

print("\n" + "="*80)
print("MODEL 2: RANDOM FOREST")
print("="*80)

# Random Forest with limited trees (small dataset)
rf_model = RandomForestClassifier(
    n_estimators=100,         # Fewer trees for small data
    max_depth=4,              # Same depth as DT
    min_samples_leaf=3,       # Same as DT
    class_weight='balanced',  # Handle imbalance
    random_state=42,
    n_jobs=-1                 # Use all cores
)

# Cross-validation
print("\nPerforming 5-fold stratified cross-validation...")

cv_results_rf = cross_validate(
    rf_model, X, y, cv=cv,
    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    return_train_score=True
)

print("\n--- Random Forest Cross-Validation Results ---")
print(f"Accuracy:  {cv_results_rf['test_accuracy'].mean():.3f} ± {cv_results_rf['test_accuracy'].std():.3f}")
print(f"Precision: {cv_results_rf['test_precision'].mean():.3f} ± {cv_results_rf['test_precision'].std():.3f}")
print(f"Recall:    {cv_results_rf['test_recall'].mean():.3f} ± {cv_results_rf['test_recall'].std():.3f}")
print(f"F1:        {cv_results_rf['test_f1'].mean():.3f} ± {cv_results_rf['test_f1'].std():.3f}")
print(f"ROC-AUC:   {cv_results_rf['test_roc_auc'].mean():.3f} ± {cv_results_rf['test_roc_auc'].std():.3f}")

# Check for overfitting
train_acc_rf = cv_results_rf['train_accuracy'].mean()
test_acc_rf = cv_results_rf['test_accuracy'].mean()
overfit_gap_rf = train_acc_rf - test_acc_rf

print(f"\nOverfitting check:")
print(f"  Train accuracy: {train_acc_rf:.3f}")
print(f"  Test accuracy:  {test_acc_rf:.3f}")
print(f"  Gap:            {overfit_gap_rf:.3f}")

if overfit_gap_rf > 0.15:
    print("  ⚠️  Model may be overfitting (gap > 0.15)")
elif overfit_gap_rf > 0.10:
    print("  ⚠️  Moderate overfitting (0.10 < gap < 0.15)")
else:
    print("  ✓ Acceptable generalization (gap < 0.10)")

# Train on full data for final model
rf_model.fit(X, y)

# Feature importances
rf_importances = pd.DataFrame({
    'feature': features_to_use,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n--- Random Forest Feature Importances ---")
print(rf_importances.to_string(index=False))

# ============================================================================
# MODEL COMPARISON
# ============================================================================

print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'Overfit Gap'],
    'Decision Tree': [
        cv_results_dt['test_accuracy'].mean(),
        cv_results_dt['test_precision'].mean(),
        cv_results_dt['test_recall'].mean(),
        cv_results_dt['test_f1'].mean(),
        cv_results_dt['test_roc_auc'].mean(),
        overfit_gap
    ],
    'Random Forest': [
        cv_results_rf['test_accuracy'].mean(),
        cv_results_rf['test_precision'].mean(),
        cv_results_rf['test_recall'].mean(),
        cv_results_rf['test_f1'].mean(),
        cv_results_rf['test_roc_auc'].mean(),
        overfit_gap_rf
    ]
})

print("\n" + comparison.to_string(index=False))

# Determine winner
if cv_results_rf['test_f1'].mean() > cv_results_dt['test_f1'].mean():
    best_model = rf_model
    best_model_name = "Random Forest"
    best_f1 = cv_results_rf['test_f1'].mean()
else:
    best_model = dt_model
    best_model_name = "Decision Tree"
    best_f1 = cv_results_dt['test_f1'].mean()

print(f"\n✓ BEST MODEL: {best_model_name} (F1 = {best_f1:.3f})")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# 1. Feature Importance Comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Decision Tree
axes[0].barh(dt_importances['feature'], dt_importances['importance'], color='steelblue', edgecolor='black')
axes[0].set_xlabel('Importance', fontweight='bold')
axes[0].set_title('Decision Tree Feature Importances', fontsize=14, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

# Random Forest
axes[1].barh(rf_importances['feature'], rf_importances['importance'], color='forestgreen', edgecolor='black')
axes[1].set_xlabel('Importance', fontweight='bold')
axes[1].set_title('Random Forest Feature Importances', fontsize=14, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
import os
os.makedirs('outputs/visualizations', exist_ok=True)
plt.savefig('outputs/visualizations/model_feature_importances.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Saved: outputs/visualizations/model_feature_importances.png")

# 2. Cross-Validation Performance Comparison
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']

for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
    dt_scores = cv_results_dt[f'test_{metric}']
    rf_scores = cv_results_rf[f'test_{metric}']
    
    axes[idx].boxplot([dt_scores, rf_scores], labels=['Decision Tree', 'Random Forest'])
    axes[idx].set_ylabel(name, fontweight='bold')
    axes[idx].set_title(f'{name} Distribution Across Folds', fontweight='bold')
    axes[idx].grid(axis='y', alpha=0.3)
    
    # Add mean lines
    axes[idx].axhline(dt_scores.mean(), color='blue', linestyle='--', alpha=0.5, label='DT Mean')
    axes[idx].axhline(rf_scores.mean(), color='green', linestyle='--', alpha=0.5, label='RF Mean')
    axes[idx].legend(fontsize=8)

# Hide the 6th subplot
axes[5].axis('off')

plt.tight_layout()
plt.savefig('outputs/visualizations/model_cv_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Saved: outputs/visualizations/model_cv_comparison.png")

# ============================================================================
# SAVE MODELS
# ============================================================================

print("\n" + "="*80)
print("SAVING MODELS")
print("="*80)

os.makedirs('models', exist_ok=True)

# Save both models with metadata
dt_package = {
    'model': dt_model,
    'model_name': 'Decision Tree',
    'features': features_to_use,
    'cv_results': {
        'accuracy': cv_results_dt['test_accuracy'].mean(),
        'precision': cv_results_dt['test_precision'].mean(),
        'recall': cv_results_dt['test_recall'].mean(),
        'f1': cv_results_dt['test_f1'].mean(),
        'roc_auc': cv_results_dt['test_roc_auc'].mean()
    },
    'feature_importances': dt_importances.to_dict(),
    'trained_on': '5k_with_proxies.csv',
    'n_samples': len(X),
    'class_distribution': y.value_counts().to_dict()
}

rf_package = {
    'model': rf_model,
    'model_name': 'Random Forest',
    'features': features_to_use,
    'cv_results': {
        'accuracy': cv_results_rf['test_accuracy'].mean(),
        'precision': cv_results_rf['test_precision'].mean(),
        'recall': cv_results_rf['test_recall'].mean(),
        'f1': cv_results_rf['test_f1'].mean(),
        'roc_auc': cv_results_rf['test_roc_auc'].mean()
    },
    'feature_importances': rf_importances.to_dict(),
    'trained_on': '5k_with_proxies.csv',
    'n_samples': len(X),
    'class_distribution': y.value_counts().to_dict()
}

joblib.dump(dt_package, 'models/decision_tree_model.pkl')
joblib.dump(rf_package, 'models/random_forest_model.pkl')

print("✓ Saved: models/decision_tree_model.pkl")
print("✓ Saved: models/random_forest_model.pkl")

# Save best model separately
joblib.dump(best_model, 'models/best_model.pkl')
print(f"✓ Saved: models/best_model.pkl ({best_model_name})")

# ============================================================================
# MILESTONE 3 SUMMARY
# ============================================================================

summary = f"""
{'='*80}
MILESTONE 3: MODEL TRAINING - COMPLETE
{'='*80}

STRATEGY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Trained WITHOUT proxy features (based on Milestone 2 validation failure)
✅ Used only {len(features_to_use)} features that work on both 5k and 2k naturally
✅ 5-fold stratified cross-validation (more reliable than single test set)
✅ Trained Decision Tree and Random Forest
✅ Compared performance across multiple metrics


FEATURES USED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chr(10).join([f"{i+1}. {feat}" for i, feat in enumerate(features_to_use)])}


DECISION TREE RESULTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accuracy:  {cv_results_dt['test_accuracy'].mean():.3f} ± {cv_results_dt['test_accuracy'].std():.3f}
Precision: {cv_results_dt['test_precision'].mean():.3f} ± {cv_results_dt['test_precision'].std():.3f}
Recall:    {cv_results_dt['test_recall'].mean():.3f} ± {cv_results_dt['test_recall'].std():.3f}
F1 Score:  {cv_results_dt['test_f1'].mean():.3f} ± {cv_results_dt['test_f1'].std():.3f}
ROC-AUC:   {cv_results_dt['test_roc_auc'].mean():.3f} ± {cv_results_dt['test_roc_auc'].std():.3f}
Overfit Gap: {overfit_gap:.3f}


RANDOM FOREST RESULTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accuracy:  {cv_results_rf['test_accuracy'].mean():.3f} ± {cv_results_rf['test_accuracy'].std():.3f}
Precision: {cv_results_rf['test_precision'].mean():.3f} ± {cv_results_rf['test_precision'].std():.3f}
Recall:    {cv_results_rf['test_recall'].mean():.3f} ± {cv_results_rf['test_recall'].std():.3f}
F1 Score:  {cv_results_rf['test_f1'].mean():.3f} ± {cv_results_rf['test_f1'].std():.3f}
ROC-AUC:   {cv_results_rf['test_roc_auc'].mean():.3f} ± {cv_results_rf['test_roc_auc'].std():.3f}
Overfit Gap: {overfit_gap_rf:.3f}


BEST MODEL: {best_model_name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
F1 Score: {best_f1:.3f}


TOP 3 PREDICTIVE FEATURES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Decision Tree:
{chr(10).join([f"  {i+1}. {row['feature']}: {row['importance']:.3f}" for i, (_, row) in enumerate(dt_importances.head(3).iterrows())])}

Random Forest:
{chr(10).join([f"  {i+1}. {row['feature']}: {row['importance']:.3f}" for i, (_, row) in enumerate(rf_importances.head(3).iterrows())])}


Our Results: {cv_results_rf['test_accuracy'].mean():.1%} accuracy, {cv_results_rf['test_precision'].mean():.2f} precision, {cv_results_rf['test_recall'].mean():.2f} recall (on 5-fold CV)

Improvements:
✅ More reliable validation (5-fold CV vs single test split)
✅ Fewer features (6 vs their full set) - better generalization
✅ No proxy features - works on 2k dataset without issues
✅ Clear feature importance from both models


NEXT MILESTONE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
→ MILESTONE 4: Business Validation on 2k Dataset
  - Apply {best_model_name} to 2k operational logs
  - Validate predictions using business metrics:
    * cost_usd correlation
    * rework_flag lift
    * sla_breach correlation
  - Assess if model generalizes to production-like data

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

print(summary)

# Save summary
with open('outputs/reports/milestone3_summary.txt', 'w') as f:
    f.write(summary)

print("\n✓ Saved: outputs/reports/milestone3_summary.txt")

print("\n" + "="*80)
print("🎉 MILESTONE 3 COMPLETE!")
print("="*80)
print(f"\nBest Model: {best_model_name}")
print(f"F1 Score: {best_f1:.3f}")
print("\nReady for MILESTONE 4: Business Validation on 2k Dataset")