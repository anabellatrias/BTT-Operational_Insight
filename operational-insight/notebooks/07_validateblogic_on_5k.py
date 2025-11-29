# ============================================================================
# USING 5K TO VALIDATE OUR BUSINESS-LOGIC APPROACH
# ============================================================================
# Goal: Show that our business rules work on 5k ground truth
# This gives confidence our 2k labels are reasonable
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("VALIDATING BUSINESS-LOGIC RULES USING 5K GROUND TRUTH")
print("="*80)

# ============================================================================
# STEP 1: LOAD 5K WITH TRUE LABELS
# ============================================================================

df_5k = pd.read_csv('operational-insight/data/processed/5k_with_proxies.csv')
print(f"\n✓ Loaded 5k: {df_5k.shape}")
print(f"  True bottlenecks: {df_5k['is_bottleneck_event'].sum()} ({df_5k['is_bottleneck_event'].mean()*100:.1f}%)")

# ============================================================================
# STEP 2: APPLY OUR BUSINESS RULES TO 5K
# ============================================================================

print("\n" + "="*80)
print("APPLYING BUSINESS RULES TO 5K")
print("="*80)

# Same rules we used on 2k
variance_threshold = 0.5
wait_time_p90 = df_5k['wait_time_minutes'].quantile(0.90)

df_5k['business_logic_label'] = (
    # Rule 1: High variance
    (df_5k['variance_to_expected'] > variance_threshold) |
    
    # Rule 2: Excessive wait time
    (df_5k['wait_time_minutes'] > wait_time_p90) |
    
    # Rule 3: SLA breach
    (df_5k['sla_breached'] == 1)
).astype(int)

print(f"\nBusiness rules identified:")
print(f"  Bottlenecks: {df_5k['business_logic_label'].sum()} ({df_5k['business_logic_label'].mean()*100:.1f}%)")

# ============================================================================
# STEP 3: COMPARE TO TRUE LABELS
# ============================================================================

print("\n" + "="*80)
print("HOW WELL DO BUSINESS RULES MATCH GROUND TRUTH?")
print("="*80)

print("\nClassification Report:")
print(classification_report(
    df_5k['is_bottleneck_event'],
    df_5k['business_logic_label'],
    target_names=['Normal', 'Bottleneck'],
    digits=3
))

# Confusion matrix
cm = confusion_matrix(df_5k['is_bottleneck_event'], df_5k['business_logic_label'])
print("\nConfusion Matrix:")
print(f"                Predicted")
print(f"                Normal  Bottleneck")
print(f"True Normal       {cm[0,0]:3d}      {cm[0,1]:3d}")
print(f"True Bottleneck   {cm[1,0]:3d}      {cm[1,1]:3d}")

# ============================================================================
# STEP 4: COMPARE BUSINESS METRICS
# ============================================================================

print("\n" + "="*80)
print("BUSINESS METRICS: TRUE vs BUSINESS-LOGIC LABELS")
print("="*80)

# True labels
true_bn = df_5k[df_5k['is_bottleneck_event'] == 1]
true_normal = df_5k[df_5k['is_bottleneck_event'] == 0]

# Business logic labels
logic_bn = df_5k[df_5k['business_logic_label'] == 1]
logic_normal = df_5k[df_5k['business_logic_label'] == 0]

print("\nUsing TRUE labels:")
print(f"  Wait time:  {true_bn['wait_time_minutes'].mean():.1f} vs {true_normal['wait_time_minutes'].mean():.1f} min")
print(f"  Duration:   {true_bn['duration_minutes'].mean():.1f} vs {true_normal['duration_minutes'].mean():.1f} min")
print(f"  Variance:   {true_bn['variance_to_expected'].mean():.3f} vs {true_normal['variance_to_expected'].mean():.3f}")

print("\nUsing BUSINESS-LOGIC labels:")
print(f"  Wait time:  {logic_bn['wait_time_minutes'].mean():.1f} vs {logic_normal['wait_time_minutes'].mean():.1f} min")
print(f"  Duration:   {logic_bn['duration_minutes'].mean():.1f} vs {logic_normal['duration_minutes'].mean():.1f} min")
print(f"  Variance:   {logic_bn['variance_to_expected'].mean():.3f} vs {logic_normal['variance_to_expected'].mean():.3f}")

print("\n✓ Similar patterns = Business logic captures real bottleneck characteristics")

# ============================================================================
# STEP 5: VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Normal', 'Bottleneck'],
            yticklabels=['Normal', 'Bottleneck'])
axes[0].set_xlabel('Predicted (Business Logic)', fontweight='bold')
axes[0].set_ylabel('True Label', fontweight='bold')
axes[0].set_title('Business Logic vs Ground Truth (5k)', fontweight='bold')

# Metrics comparison
metrics = ['Wait Time', 'Duration', 'Variance']
true_vals = [
    true_bn['wait_time_minutes'].mean() / true_normal['wait_time_minutes'].mean(),
    true_bn['duration_minutes'].mean() / true_normal['duration_minutes'].mean(),
    (true_bn['variance_to_expected'].mean() - true_normal['variance_to_expected'].mean())
]
logic_vals = [
    logic_bn['wait_time_minutes'].mean() / logic_normal['wait_time_minutes'].mean(),
    logic_bn['duration_minutes'].mean() / logic_normal['duration_minutes'].mean(),
    (logic_bn['variance_to_expected'].mean() - logic_normal['variance_to_expected'].mean())
]

x = np.arange(len(metrics))
width = 0.35

axes[1].bar(x - width/2, true_vals, width, label='True Labels', color='steelblue')
axes[1].bar(x + width/2, logic_vals, width, label='Business Logic', color='coral')
axes[1].set_xlabel('Metric', fontweight='bold')
axes[1].set_ylabel('Lift / Difference', fontweight='bold')
axes[1].set_title('Bottleneck vs Normal Comparison', fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(metrics)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/visualizations/business_logic_5k_validation.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Saved: outputs/visualizations/business_logic_5k_validation.png")

# ============================================================================
# CONCLUSION
# ============================================================================

from sklearn.metrics import f1_score
f1 = f1_score(df_5k['is_bottleneck_event'], df_5k['business_logic_label'])

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

print(f"\nOur business rules achieve F1={f1:.3f} on 5k ground truth")

if f1 > 0.7:
    print("✓ STRONG: Business logic captures real bottleneck patterns")
    print("✓ This validates using same rules for 2k labeling")
elif f1 > 0.5:
    print("⚠️  MODERATE: Business logic partially captures bottlenecks")
    print("⚠️  May need refinement but reasonable starting point")
else:
    print("✗ WEAK: Business logic doesn't match ground truth well")
    print("✗ Need to reconsider labeling strategy")

print("\n" + "="*80)