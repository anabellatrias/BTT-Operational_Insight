# ============================================================================
# MILESTONE 4: BUSINESS VALIDATION ON 2K DATASET
# ============================================================================
# Goal: Validate that our model's predictions correlate with business outcomes
# Strategy: Apply Random Forest to 2k, check correlation with cost/rework/SLA
# ============================================================================

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MILESTONE 4: BUSINESS VALIDATION")
print("="*80)

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================

print("\n--- Loading Random Forest Model ---")
rf_package = joblib.load('models/random_forest_model.pkl')
model = rf_package['model']
feature_names = rf_package['features']

print(f"✓ Loaded Random Forest model")
print(f"✓ Features: {feature_names}")
print(f"✓ Training F1: {rf_package['cv_results']['f1']:.3f}")

print("\n--- Loading 2k Processed Dataset ---")
df_2k = pd.read_csv('operational-insight/data/processed/2k_processed.csv')
print(f"✓ Loaded 2k dataset: {df_2k.shape}")

# ============================================================================
# PREPARE FEATURES FOR PREDICTION
# ============================================================================

print("\n" + "="*80)
print("FEATURE PREPARATION")
print("="*80)

# Check if all required features exist
missing_features = []
for feat in feature_names:
    if feat not in df_2k.columns:
        missing_features.append(feat)
        print(f"  ❌ {feat} - MISSING")
    else:
        print(f"  ✓ {feat}")

if missing_features:
    print(f"\n⚠️  WARNING: Missing {len(missing_features)} features!")
    print("Cannot proceed with predictions.")
else:
    print(f"\n✓ All {len(feature_names)} features available")

# Create feature matrix
X_2k = df_2k[feature_names].copy()

# Handle missing values (same strategy as training)
X_2k = X_2k.fillna(X_2k.median())

print(f"\n✓ Feature matrix prepared: {X_2k.shape}")
print("\nFeature statistics:")
print(X_2k.describe())

# ============================================================================
# MAKE PREDICTIONS
# ============================================================================

print("\n" + "="*80)
print("GENERATING PREDICTIONS")
print("="*80)

# Predict bottleneck probability and class
df_2k['bottleneck_probability'] = model.predict_proba(X_2k)[:, 1]
df_2k['bottleneck_prediction'] = model.predict(X_2k)

print(f"✓ Generated predictions for {len(df_2k)} events")

# Prediction distribution
pred_counts = df_2k['bottleneck_prediction'].value_counts().sort_index()
print(f"\nPrediction Distribution:")
print(f"  Normal (0):     {pred_counts[0]:,} ({pred_counts[0]/len(df_2k)*100:.1f}%)")
print(f"  Bottleneck (1): {pred_counts[1]:,} ({pred_counts[1]/len(df_2k)*100:.1f}%)")

# Probability distribution
print(f"\nBottleneck Probability Statistics:")
print(f"  Mean:   {df_2k['bottleneck_probability'].mean():.3f}")
print(f"  Median: {df_2k['bottleneck_probability'].median():.3f}")
print(f"  Std:    {df_2k['bottleneck_probability'].std():.3f}")
print(f"  Min:    {df_2k['bottleneck_probability'].min():.3f}")
print(f"  Max:    {df_2k['bottleneck_probability'].max():.3f}")

# ============================================================================
# BUSINESS VALIDATION - CORE TESTS
# ============================================================================

print("\n" + "="*80)
print("BUSINESS VALIDATION TESTS")
print("="*80)

# Separate predicted bottlenecks vs normal
predicted_bottlenecks = df_2k[df_2k['bottleneck_prediction'] == 1]
predicted_normal = df_2k[df_2k['bottleneck_prediction'] == 0]

validation_results = {}

# ----------------------------------------------------------------------------
# TEST 1: Cost Analysis
# ----------------------------------------------------------------------------

print("\n" + "-"*80)
print("TEST 1: COST CORRELATION")
print("-"*80)

cost_bottleneck = predicted_bottlenecks['cost_usd'].mean()
cost_normal = predicted_normal['cost_usd'].mean()
cost_lift = cost_bottleneck / cost_normal if cost_normal > 0 else np.inf

# Statistical test
t_stat_cost, p_value_cost = stats.ttest_ind(
    predicted_bottlenecks['cost_usd'].dropna(),
    predicted_normal['cost_usd'].dropna()
)

# Correlation
corr_cost, p_corr_cost = stats.pearsonr(
    df_2k['bottleneck_probability'],
    df_2k['cost_usd']
)

print(f"Average cost_usd:")
print(f"  Predicted bottlenecks: ${cost_bottleneck:.2f}")
print(f"  Predicted normal:      ${cost_normal:.2f}")
print(f"  Lift:                  {cost_lift:.2f}x")
print(f"\nStatistical test:")
print(f"  T-statistic: {t_stat_cost:.3f}")
print(f"  P-value:     {p_value_cost:.4f}")
print(f"\nCorrelation:")
print(f"  r = {corr_cost:.3f}, p = {p_corr_cost:.4f}")

if cost_lift > 1.5 and p_value_cost < 0.05:
    print("  ✓ STRONG VALIDATION: Bottlenecks have significantly higher costs")
    validation_results['cost'] = 'PASS'
elif cost_lift > 1.2:
    print("  ⚠️  MODERATE VALIDATION: Some cost difference detected")
    validation_results['cost'] = 'WEAK'
else:
    print("  ❌ WEAK VALIDATION: Little cost difference")
    validation_results['cost'] = 'FAIL'

# ----------------------------------------------------------------------------
# TEST 2: Rework Analysis
# ----------------------------------------------------------------------------

print("\n" + "-"*80)
print("TEST 2: REWORK FLAG CORRELATION")
print("-"*80)

rework_rate_bottleneck = predicted_bottlenecks['rework_flag'].mean()
rework_rate_normal = predicted_normal['rework_flag'].mean()
rework_lift = rework_rate_bottleneck / rework_rate_normal if rework_rate_normal > 0 else np.inf

# Chi-square test for categorical association
from scipy.stats import chi2_contingency

contingency_table = pd.crosstab(
    df_2k['bottleneck_prediction'],
    df_2k['rework_flag']
)
chi2, p_chi2, dof, expected = chi2_contingency(contingency_table)

print(f"Rework rate:")
print(f"  Predicted bottlenecks: {rework_rate_bottleneck:.1%}")
print(f"  Predicted normal:      {rework_rate_normal:.1%}")
print(f"  Lift:                  {rework_lift:.2f}x")
print(f"\nChi-square test:")
print(f"  χ² = {chi2:.3f}, p = {p_chi2:.4f}")

if rework_lift > 2.0 and p_chi2 < 0.05:
    print("  ✓ STRONG VALIDATION: Bottlenecks have significantly more rework")
    validation_results['rework'] = 'PASS'
elif rework_lift > 1.5:
    print("  ⚠️  MODERATE VALIDATION: Some rework association")
    validation_results['rework'] = 'WEAK'
else:
    print("  ❌ WEAK VALIDATION: Little rework difference")
    validation_results['rework'] = 'FAIL'

# ----------------------------------------------------------------------------
# TEST 3: SLA Breach Analysis
# ----------------------------------------------------------------------------

print("\n" + "-"*80)
print("TEST 3: SLA BREACH CORRELATION")
print("-"*80)

sla_rate_bottleneck = predicted_bottlenecks['sla_breached'].mean()
sla_rate_normal = predicted_normal['sla_breached'].mean()
sla_lift = sla_rate_bottleneck / sla_rate_normal if sla_rate_normal > 0 else np.inf

# Chi-square test
contingency_table_sla = pd.crosstab(
    df_2k['bottleneck_prediction'],
    df_2k['sla_breached']
)
chi2_sla, p_chi2_sla, dof_sla, expected_sla = chi2_contingency(contingency_table_sla)

print(f"SLA breach rate:")
print(f"  Predicted bottlenecks: {sla_rate_bottleneck:.1%}")
print(f"  Predicted normal:      {sla_rate_normal:.1%}")
print(f"  Lift:                  {sla_lift:.2f}x")
print(f"\nChi-square test:")
print(f"  χ² = {chi2_sla:.3f}, p = {p_chi2_sla:.4f}")

if sla_lift > 1.5 and p_chi2_sla < 0.05:
    print("  ✓ STRONG VALIDATION: Bottlenecks have significantly more SLA breaches")
    validation_results['sla'] = 'PASS'
elif sla_lift > 1.2:
    print("  ⚠️  MODERATE VALIDATION: Some SLA breach association")
    validation_results['sla'] = 'WEAK'
else:
    print("  ❌ WEAK VALIDATION: Little SLA breach difference")
    validation_results['sla'] = 'FAIL'

# ----------------------------------------------------------------------------
# TEST 4: Wait Time & Duration Validation
# ----------------------------------------------------------------------------

print("\n" + "-"*80)
print("TEST 4: OPERATIONAL METRICS VALIDATION")
print("-"*80)

wait_bottleneck = predicted_bottlenecks['wait_time_minutes'].mean()
wait_normal = predicted_normal['wait_time_minutes'].mean()
wait_lift = wait_bottleneck / wait_normal if wait_normal > 0 else np.inf

duration_bottleneck = predicted_bottlenecks['duration_minutes'].mean()
duration_normal = predicted_normal['duration_minutes'].mean()
duration_lift = duration_bottleneck / duration_normal if duration_normal > 0 else np.inf

print(f"Wait time:")
print(f"  Predicted bottlenecks: {wait_bottleneck:.2f} min")
print(f"  Predicted normal:      {wait_normal:.2f} min")
print(f"  Lift:                  {wait_lift:.2f}x")

print(f"\nDuration:")
print(f"  Predicted bottlenecks: {duration_bottleneck:.2f} min")
print(f"  Predicted normal:      {duration_normal:.2f} min")
print(f"  Lift:                  {duration_lift:.2f}x")

if wait_lift > 1.3 and duration_lift > 1.3:
    print("  ✓ VALIDATION: Predicted bottlenecks show expected operational patterns")
    validation_results['operational'] = 'PASS'
else:
    print("  ⚠️  WARNING: Operational metrics don't strongly differentiate")
    validation_results['operational'] = 'WEAK'

# ============================================================================
# OVERALL VALIDATION ASSESSMENT
# ============================================================================

print("\n" + "="*80)
print("OVERALL VALIDATION ASSESSMENT")
print("="*80)

print("\nValidation Results Summary:")
for test, result in validation_results.items():
    symbol = "✓" if result == "PASS" else "⚠️" if result == "WEAK" else "❌"
    print(f"  {symbol} {test.upper()}: {result}")

passes = sum(1 for r in validation_results.values() if r == "PASS")
total = len(validation_results)

print(f"\nPassed: {passes}/{total} tests")

if passes >= 3:
    overall = "STRONG"
    print("\n✅ STRONG VALIDATION: Model predictions correlate well with business outcomes")
    print("   → Recommended: DEPLOY with confidence")
elif passes >= 2:
    overall = "MODERATE"
    print("\n⚠️  MODERATE VALIDATION: Some business correlation detected")
    print("   → Recommended: Deploy with monitoring and refinement plan")
else:
    overall = "WEAK"
    print("\n❌ WEAK VALIDATION: Limited business correlation")
    print("   → Recommended: Do NOT deploy, investigate further")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("CREATING VALIDATION VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Bottleneck Probability Distribution
ax1 = fig.add_subplot(gs[0, :])
ax1.hist(df_2k['bottleneck_probability'], bins=50, edgecolor='black', alpha=0.7)
ax1.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
ax1.set_xlabel('Bottleneck Probability', fontweight='bold')
ax1.set_ylabel('Frequency', fontweight='bold')
ax1.set_title('Distribution of Bottleneck Probabilities on 2k Dataset', 
              fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Cost Comparison
ax2 = fig.add_subplot(gs[1, 0])
cost_data = [predicted_normal['cost_usd'], predicted_bottlenecks['cost_usd']]
bp = ax2.boxplot(cost_data, labels=['Normal', 'Bottleneck'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightgreen')
bp['boxes'][1].set_facecolor('lightcoral')
ax2.set_ylabel('Cost (USD)', fontweight='bold')
ax2.set_title(f'Cost Comparison\nLift: {cost_lift:.2f}x', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# 3. Rework Rate Comparison
ax3 = fig.add_subplot(gs[1, 1])
rework_rates = [rework_rate_normal * 100, rework_rate_bottleneck * 100]
bars = ax3.bar(['Normal', 'Bottleneck'], rework_rates, 
               color=['lightgreen', 'lightcoral'], edgecolor='black')
ax3.set_ylabel('Rework Rate (%)', fontweight='bold')
ax3.set_title(f'Rework Flag Rate\nLift: {rework_lift:.2f}x', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
for bar, rate in zip(bars, rework_rates):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{rate:.1f}%', ha='center', fontweight='bold')

# 4. SLA Breach Rate Comparison
ax4 = fig.add_subplot(gs[1, 2])
sla_rates = [sla_rate_normal * 100, sla_rate_bottleneck * 100]
bars = ax4.bar(['Normal', 'Bottleneck'], sla_rates,
               color=['lightgreen', 'lightcoral'], edgecolor='black')
ax4.set_ylabel('SLA Breach Rate (%)', fontweight='bold')
ax4.set_title(f'SLA Breach Rate\nLift: {sla_lift:.2f}x', fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
for bar, rate in zip(bars, sla_rates):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{rate:.1f}%', ha='center', fontweight='bold')

# 5. Wait Time Comparison
ax5 = fig.add_subplot(gs[2, 0])
wait_data = [predicted_normal['wait_time_minutes'], predicted_bottlenecks['wait_time_minutes']]
bp = ax5.boxplot(wait_data, labels=['Normal', 'Bottleneck'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightgreen')
bp['boxes'][1].set_facecolor('lightcoral')
ax5.set_ylabel('Wait Time (minutes)', fontweight='bold')
ax5.set_title(f'Wait Time Comparison\nLift: {wait_lift:.2f}x', fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

# 6. Duration Comparison
ax6 = fig.add_subplot(gs[2, 1])
duration_data = [predicted_normal['duration_minutes'], predicted_bottlenecks['duration_minutes']]
bp = ax6.boxplot(duration_data, labels=['Normal', 'Bottleneck'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightgreen')
bp['boxes'][1].set_facecolor('lightcoral')
ax6.set_ylabel('Duration (minutes)', fontweight='bold')
ax6.set_title(f'Duration Comparison\nLift: {duration_lift:.2f}x', fontweight='bold')
ax6.grid(axis='y', alpha=0.3)

# 7. Validation Summary
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')
summary_text = f"""
BUSINESS VALIDATION SUMMARY

Tests Passed: {passes}/{total}

Results:
- Cost Lift: {cost_lift:.2f}x
- Rework Lift: {rework_lift:.2f}x  
- SLA Lift: {sla_lift:.2f}x
- Wait Time Lift: {wait_lift:.2f}x
- Duration Lift: {duration_lift:.2f}x

Overall: {overall}

Model F1: {rf_package['cv_results']['f1']:.3f}
"""
ax7.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
         verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle('Business Validation: Random Forest on 2k Dataset', 
             fontsize=16, fontweight='bold', y=0.98)

import os
os.makedirs('outputs/visualizations', exist_ok=True)
plt.savefig('outputs/visualizations/business_validation.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Saved: outputs/visualizations/business_validation.png")

# ============================================================================
# TOP PREDICTED BOTTLENECKS
# ============================================================================

print("\n" + "="*80)
print("TOP 10 PREDICTED BOTTLENECKS")
print("="*80)

top_bottlenecks = df_2k.nlargest(10, 'bottleneck_probability')[[
    'case_id', 'activity_name', 'bottleneck_probability', 
    'wait_time_minutes', 'duration_minutes', 'cost_usd', 
    'rework_flag', 'sla_breached'
]]

print("\n" + top_bottlenecks.to_string(index=False))

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING VALIDATION RESULTS")
print("="*80)

# Save predictions
df_2k.to_csv('operational-insight/data/processed/2k_with_predictions.csv', index=False)
print("✓ Saved: operational-insight/data/processed/2k_with_predictions.csv")

# Create validation report
report = f"""
{'='*80}
MILESTONE 4: BUSINESS VALIDATION - COMPLETE
{'='*80}

MODEL PERFORMANCE (from training):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model: Random Forest
Training F1: {rf_package['cv_results']['f1']:.3f}
Training Recall: {rf_package['cv_results']['recall']:.3f}
Training Precision: {rf_package['cv_results']['precision']:.3f}


PREDICTIONS ON 2K DATASET:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total events: {len(df_2k):,}
Predicted normal: {pred_counts[0]:,} ({pred_counts[0]/len(df_2k)*100:.1f}%)
Predicted bottlenecks: {pred_counts[1]:,} ({pred_counts[1]/len(df_2k)*100:.1f}%)


BUSINESS VALIDATION RESULTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST 1: Cost Correlation
  Predicted bottleneck cost: ${cost_bottleneck:.2f}
  Predicted normal cost: ${cost_normal:.2f}
  Lift: {cost_lift:.2f}x
  P-value: {p_value_cost:.4f}
  Result: {validation_results['cost']}

TEST 2: Rework Flag
  Bottleneck rework rate: {rework_rate_bottleneck:.1%}
  Normal rework rate: {rework_rate_normal:.1%}
  Lift: {rework_lift:.2f}x
  P-value: {p_chi2:.4f}
  Result: {validation_results['rework']}

TEST 3: SLA Breach
  Bottleneck SLA breach rate: {sla_rate_bottleneck:.1%}
  Normal SLA breach rate: {sla_rate_normal:.1%}
  Lift: {sla_lift:.2f}x
  P-value: {p_chi2_sla:.4f}
  Result: {validation_results['sla']}

TEST 4: Operational Metrics
  Wait time lift: {wait_lift:.2f}x
  Duration lift: {duration_lift:.2f}x
  Result: {validation_results['operational']}


OVERALL ASSESSMENT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tests passed: {passes}/{total}
Validation strength: {overall}

{'✅ Model predictions correlate well with business outcomes' if overall == 'STRONG' else '⚠️ Model shows some business correlation' if overall == 'MODERATE' else '❌ Model shows weak business correlation'}


INTERPRETATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The model was trained on 5k labeled data and applied to 2k operational logs.
Despite different data sources and activity granularity, the model's predictions
show {'strong' if overall == 'STRONG' else 'moderate' if overall == 'MODERATE' else 'weak'} correlation with business metrics (cost, rework, SLA breaches).

This {'validates' if overall == 'STRONG' else 'partially validates' if overall == 'MODERATE' else 'questions'} that the model has learned generalizable bottleneck patterns.


RECOMMENDATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{'✅ DEPLOY: Model is ready for production use with monitoring' if overall == 'STRONG' else '⚠️ DEPLOY WITH CAUTION: Monitor predictions and refine as needed' if overall == 'MODERATE' else '❌ DO NOT DEPLOY: Investigate model limitations before production use'}


Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open('outputs/reports/milestone4_validation_report.txt', 'w') as f:
    f.write(report)

print("✓ Saved: outputs/reports/milestone4_validation_report.txt")

print(report)

print("\n" + "="*80)
print("🎉 MILESTONE 4 COMPLETE!")
print("="*80)
print(f"\nOverall Validation: {overall}")
print(f"Recommendation: {'DEPLOY' if overall in ['STRONG', 'MODERATE'] else 'DO NOT DEPLOY'}")