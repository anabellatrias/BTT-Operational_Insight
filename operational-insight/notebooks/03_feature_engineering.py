# ============================================================================
# MILESTONE 2: FEATURE ENGINEERING
# ============================================================================
# Goal: Create standardized features that work on both 5k and 2k datasets
# Strategy: Hybrid approach for activity mismatch + proxy features
# ============================================================================

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

print("="*80)
print("MILESTONE 2: FEATURE ENGINEERING")
print("="*80)

# ============================================================================
# LOAD DATASETS
# ============================================================================

from pathlib import Path
# Move up to repo root so `data/` resolves no matter where this notebook lives
repo_dir = Path(__file__).resolve().parents[1]
data_path = repo_dir / "data" / "raw"

df_5k = pd.read_csv(data_path / "5kp.csv")
df_2k = pd.read_csv(data_path / "2k.csv")
df_activity_ref = pd.read_csv(data_path / "Activity_Reference.csv")

# Clean up
if 'Unnamed: 0' in df_5k.columns:
    df_5k = df_5k.drop('Unnamed: 0', axis=1)
if 'Unnamed: 0' in df_2k.columns:
    df_2k = df_2k.drop('Unnamed: 0', axis=1)

# Convert timestamps
df_5k['start_timestamp_utc'] = pd.to_datetime(df_5k['start_timestamp_utc'])
df_5k['end_timestamp_utc'] = pd.to_datetime(df_5k['end_timestamp_utc'])
df_2k['timestamp'] = pd.to_datetime(df_2k['timestamp'])

print(f"\n✓ Loaded 5k: {df_5k.shape}")
print(f"✓ Loaded 2k: {df_2k.shape}")
print(f"✓ Loaded Activity Reference: {df_activity_ref.shape}")

# ============================================================================
# STEP 2.1: COMMON FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*80)
print("STEP 2.1: COMMON FEATURE ENGINEERING")
print("="*80)

# ----------------------------------------------------------------------------
# 2.1.1: Standardize 2k dataset to match 5k structure
# ----------------------------------------------------------------------------

print("\n--- Converting 2k features to match 5k structure ---")

# Create working copy
df_2k_processed = df_2k.copy()

# Convert units (seconds to minutes)
df_2k_processed['duration_minutes'] = df_2k_processed['duration_sec'] / 60
df_2k_processed['wait_time_minutes'] = df_2k_processed['wait_time_sec'] / 60

print(f"✓ Converted duration_sec → duration_minutes")
print(f"✓ Converted wait_time_sec → wait_time_minutes")

# Rename columns for consistency
df_2k_processed = df_2k_processed.rename(columns={
    'event_nr': 'event_num_in_case',
    'activity': 'activity_name',
    'hour': 'hour_of_day',
    'sla_breach': 'sla_breached'
})

print(f"✓ Renamed columns to match 5k structure")

# ----------------------------------------------------------------------------
# 2.1.2: Calculate handoff_count_so_far for 2k
# ----------------------------------------------------------------------------

print("\n--- Calculating handoff_count_so_far for 2k ---")

def calculate_handoffs(df):
    """
    Count cumulative handoffs (resource changes) per case
    """
    df = df.sort_values(['case_id', 'timestamp']).copy()
    
    # Detect when resource changes within a case
    df['resource_changed'] = (
        df.groupby('case_id')['resource']
        .shift(1) != df['resource']
    ).astype(int)
    
    # First event in each case is not a handoff
    df.loc[df.groupby('case_id').head(1).index, 'resource_changed'] = 0
    
    # Cumulative sum of handoffs per case
    df['handoff_count_so_far'] = df.groupby('case_id')['resource_changed'].cumsum()
    
    df = df.drop('resource_changed', axis=1)
    
    return df

df_2k_processed = calculate_handoffs(df_2k_processed)

print(f"✓ Calculated handoff_count_so_far")
print(f"  Average handoffs in 2k: {df_2k_processed['handoff_count_so_far'].mean():.2f}")
print(f"  Average handoffs in 5k: {df_5k['handoff_count_so_far'].mean():.2f}")

# ============================================================================
# STEP 2.2: HANDLE ACTIVITY MISMATCH - HYBRID APPROACH
# ============================================================================

print("\n" + "="*80)
print("STEP 2.2: ACTIVITY MISMATCH - HYBRID EXPECTED DURATION")
print("="*80)

# ----------------------------------------------------------------------------
# 2.2.1: Identify which activities are in reference table
# ----------------------------------------------------------------------------

activities_in_ref = set(df_activity_ref['activity_name'].unique())
activities_5k = set(df_5k['activity_name'].unique())
activities_2k = set(df_2k_processed['activity_name'].unique())

matched_2k = activities_2k.intersection(activities_in_ref)
unmatched_2k = activities_2k - activities_in_ref

print(f"\n2k Activities IN reference table: {len(matched_2k)}")
for act in sorted(matched_2k):
    print(f"  ✓ {act}")

print(f"\n2k Activities NOT in reference table: {len(unmatched_2k)}")
for act in sorted(unmatched_2k):
    print(f"  ⚠️  {act}")

# ----------------------------------------------------------------------------
# 2.2.2: Calculate data-driven expected durations for unmatched activities
# ----------------------------------------------------------------------------

print("\n--- Calculating data-driven expected durations ---")

# Calculate median duration for each activity in 2k
activity_stats_2k = df_2k_processed.groupby('activity_name')['duration_minutes'].agg([
    ('expected_duration_minutes', 'median'),
    ('std_duration', 'std'),
    ('count', 'count')
]).reset_index()

print("\nActivity statistics from 2k data:")
print(activity_stats_2k.to_string(index=False))

# ----------------------------------------------------------------------------
# 2.2.3: Create comprehensive expected duration lookup
# ----------------------------------------------------------------------------

print("\n--- Creating comprehensive expected duration lookup ---")

# Start with reference table
expected_duration_lookup = df_activity_ref[['activity_name', 'expected_duration_minutes']].copy()
expected_duration_lookup['source'] = 'reference_table'

# Add data-driven expectations for unmatched activities
for activity in unmatched_2k:
    expected = activity_stats_2k[
        activity_stats_2k['activity_name'] == activity
    ]['expected_duration_minutes'].values[0]
    
    new_row = pd.DataFrame({
        'activity_name': [activity],
        'expected_duration_minutes': [expected],
        'source': ['data_driven']
    })
    expected_duration_lookup = pd.concat([expected_duration_lookup, new_row], ignore_index=True)

print("\nComplete expected duration lookup:")
print(expected_duration_lookup.to_string(index=False))

# Save lookup table
import os
os.makedirs('operational-insight/data/processed', exist_ok=True)
expected_duration_lookup.to_csv('operational-insight/data/processed/expected_duration_lookup.csv', index=False)
print("\n✓ Saved: operational-insight/data/processed/expected_duration_lookup.csv")

# ----------------------------------------------------------------------------
# 2.2.4: Add expected_duration_minutes to both datasets
# ----------------------------------------------------------------------------

print("\n--- Adding expected_duration to datasets ---")

# For 5k: Should already have it, but verify
if 'expected_duration_minutes' not in df_5k.columns:
    df_5k = df_5k.merge(
        expected_duration_lookup[['activity_name', 'expected_duration_minutes']],
        on='activity_name',
        how='left'
    )
    print("✓ Added expected_duration_minutes to 5k")
else:
    print("✓ 5k already has expected_duration_minutes")

# For 2k: Add from lookup
df_2k_processed = df_2k_processed.merge(
    expected_duration_lookup[['activity_name', 'expected_duration_minutes', 'source']],
    on='activity_name',
    how='left',
    suffixes=('', '_lookup')
)

# Rename source column to be clear
df_2k_processed = df_2k_processed.rename(columns={'source': 'expected_duration_source'})

print(f"✓ Added expected_duration_minutes to 2k")

# Check coverage
missing_expected = df_2k_processed['expected_duration_minutes'].isna().sum()
print(f"\nEvents missing expected_duration in 2k: {missing_expected}/{len(df_2k_processed)}")

if missing_expected > 0:
    print("⚠️  Warning: Some activities have no expected duration")
    print(df_2k_processed[df_2k_processed['expected_duration_minutes'].isna()]['activity_name'].unique())

# ----------------------------------------------------------------------------
# 2.2.5: Calculate variance_to_expected
# ----------------------------------------------------------------------------

print("\n--- Calculating variance_to_expected (THE CRITICAL FEATURE!) ---")

def calculate_variance(df):
    """
    Calculate variance_to_expected = (actual - expected) / expected
    """
    df = df.copy()
    
    # Avoid division by zero
    mask = df['expected_duration_minutes'] > 0
    
    df.loc[mask, 'variance_to_expected'] = (
        (df.loc[mask, 'duration_minutes'] - df.loc[mask, 'expected_duration_minutes']) / 
        df.loc[mask, 'expected_duration_minutes']
    )
    
    # For zero expected duration, set variance to 0
    df.loc[~mask, 'variance_to_expected'] = 0
    
    return df

df_2k_processed = calculate_variance(df_2k_processed)

print(f"✓ Calculated variance_to_expected for 2k")
print(f"\n2k Variance Statistics:")
print(f"  Mean: {df_2k_processed['variance_to_expected'].mean():.3f}")
print(f"  Median: {df_2k_processed['variance_to_expected'].median():.3f}")
print(f"  Std: {df_2k_processed['variance_to_expected'].std():.3f}")
print(f"  Min: {df_2k_processed['variance_to_expected'].min():.3f}")
print(f"  Max: {df_2k_processed['variance_to_expected'].max():.3f}")

print(f"\n5k Variance Statistics (for comparison):")
print(f"  Mean: {df_5k['variance_to_expected'].mean():.3f}")
print(f"  Median: {df_5k['variance_to_expected'].median():.3f}")
print(f"  Std: {df_5k['variance_to_expected'].std():.3f}")

# ============================================================================
# PROXY FEATURES FOR MISSING COLUMNS
# ============================================================================

print("\n" + "="*80)
print("PROXY FEATURE ENGINEERING")
print("="*80)

print("\n--- Proxy 1: queue_length_at_start → concurrent_case_count ---")

def calculate_concurrent_cases(df):
    """
    For each event, count how many other cases were active at that time
    Active = case started but not completed
    """
    df = df.sort_values('timestamp').copy()
    
    # Get case start and end times
    case_times = df.groupby('case_id')['timestamp'].agg(['min', 'max']).reset_index()
    case_times.columns = ['case_id', 'case_start', 'case_end']
    
    concurrent_counts = []
    
    for idx, row in df.iterrows():
        event_time = row['timestamp']
        event_case = row['case_id']
        
        # Count other active cases (excluding this case)
        concurrent = case_times[
            (case_times['case_id'] != event_case) &
            (case_times['case_start'] <= event_time) &
            (case_times['case_end'] >= event_time)
        ].shape[0]
        
        concurrent_counts.append(concurrent)
    
    df['concurrent_case_count'] = concurrent_counts
    
    return df

print("Calculating concurrent cases for 2k... (this may take a moment)")
df_2k_processed = calculate_concurrent_cases(df_2k_processed)

print(f"✓ Calculated concurrent_case_count")
print(f"  Mean: {df_2k_processed['concurrent_case_count'].mean():.2f}")
print(f"  Median: {df_2k_processed['concurrent_case_count'].median():.0f}")
print(f"  Max: {df_2k_processed['concurrent_case_count'].max():.0f}")

# ----------------------------------------------------------------------------
# 2.3.2: Proxy for system_load_index_0to1
# ----------------------------------------------------------------------------

print("\n--- Proxy 2: system_load_index → rolling_wait_time_normalized ---")

def calculate_system_load_proxy(df, window=10):
    """
    Calculate rolling average of wait_time, normalized to 0-1
    """
    df = df.sort_values('timestamp').copy()
    
    # Calculate rolling mean
    df['rolling_wait_time'] = df['wait_time_minutes'].rolling(
        window=window, 
        min_periods=1
    ).mean()
    
    # Normalize to 0-1
    min_val = df['rolling_wait_time'].min()
    max_val = df['rolling_wait_time'].max()
    
    if max_val > min_val:
        df['estimated_system_load'] = (df['rolling_wait_time'] - min_val) / (max_val - min_val)
    else:
        df['estimated_system_load'] = 0.5  # Default if no variance
    
    return df

df_2k_processed = calculate_system_load_proxy(df_2k_processed)

print(f"✓ Calculated estimated_system_load")
print(f"  Mean: {df_2k_processed['estimated_system_load'].mean():.3f}")
print(f"  Median: {df_2k_processed['estimated_system_load'].median():.3f}")
print(f"  Range: [{df_2k_processed['estimated_system_load'].min():.3f}, {df_2k_processed['estimated_system_load'].max():.3f}]")

# ============================================================================
# STEP 2.4: VALIDATE PROXY FEATURES ON 5K
# ============================================================================

print("\n" + "="*80)
print("STEP 2.4: VALIDATE PROXY FEATURES")
print("="*80)

print("\n--- Creating proxy features on 5k to test correlation ---")

# Create proxies for 5k using same logic
df_5k_with_proxies = df_5k.copy()

# For 5k, we need timestamps for concurrent case calculation
# Use start_timestamp_utc
df_5k_with_proxies = df_5k_with_proxies.rename(columns={'start_timestamp_utc': 'timestamp'})

# Calculate proxy for queue length
print("\nCalculating concurrent cases for 5k...")
df_5k_with_proxies = calculate_concurrent_cases(df_5k_with_proxies)

# Calculate proxy for system load
df_5k_with_proxies = calculate_system_load_proxy(df_5k_with_proxies)

# Rename back
df_5k_with_proxies = df_5k_with_proxies.rename(columns={'timestamp': 'start_timestamp_utc'})

# Now correlate proxies with actual values
print("\n" + "-"*80)
print("PROXY VALIDATION RESULTS")
print("-"*80)

# Queue length correlation
corr_queue, p_queue = stats.pearsonr(
    df_5k_with_proxies['queue_length_at_start'].dropna(),
    df_5k_with_proxies['concurrent_case_count'].dropna()
)

print(f"\nQueue Length Proxy:")
print(f"  Actual (queue_length_at_start) mean: {df_5k_with_proxies['queue_length_at_start'].mean():.2f}")
print(f"  Proxy (concurrent_case_count) mean:  {df_5k_with_proxies['concurrent_case_count'].mean():.2f}")
print(f"  Correlation: r = {corr_queue:.3f}, p = {p_queue:.4f}")

if abs(corr_queue) > 0.5:
    print(f"  ✓ GOOD proxy (|r| > 0.5)")
elif abs(corr_queue) > 0.3:
    print(f"  ⚠️  MODERATE proxy (0.3 < |r| < 0.5)")
else:
    print(f"  ❌ WEAK proxy (|r| < 0.3)")

# System load correlation
corr_load, p_load = stats.pearsonr(
    df_5k_with_proxies['system_load_index_0to1'].dropna(),
    df_5k_with_proxies['estimated_system_load'].dropna()
)

print(f"\nSystem Load Proxy:")
print(f"  Actual (system_load_index) mean: {df_5k_with_proxies['system_load_index_0to1'].mean():.3f}")
print(f"  Proxy (estimated_system_load) mean: {df_5k_with_proxies['estimated_system_load'].mean():.3f}")
print(f"  Correlation: r = {corr_load:.3f}, p = {p_load:.4f}")

if abs(corr_load) > 0.5:
    print(f"  ✓ GOOD proxy (|r| > 0.5)")
elif abs(corr_load) > 0.3:
    print(f"  ⚠️  MODERATE proxy (0.3 < |r| < 0.5)")
else:
    print(f"  ❌ WEAK proxy (|r| < 0.3)")

# Visualize correlations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Queue length scatter
axes[0].scatter(
    df_5k_with_proxies['queue_length_at_start'],
    df_5k_with_proxies['concurrent_case_count'],
    alpha=0.6, edgecolor='black'
)
axes[0].plot([0, 10], [0, 10], 'r--', label='Perfect correlation')
axes[0].set_xlabel('Actual queue_length_at_start', fontweight='bold')
axes[0].set_ylabel('Proxy: concurrent_case_count', fontweight='bold')
axes[0].set_title(f'Queue Length Proxy Validation\nr = {corr_queue:.3f}', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# System load scatter
axes[1].scatter(
    df_5k_with_proxies['system_load_index_0to1'],
    df_5k_with_proxies['estimated_system_load'],
    alpha=0.6, edgecolor='black'
)
axes[1].plot([0, 1], [0, 1], 'r--', label='Perfect correlation')
axes[1].set_xlabel('Actual system_load_index_0to1', fontweight='bold')
axes[1].set_ylabel('Proxy: estimated_system_load', fontweight='bold')
axes[1].set_title(f'System Load Proxy Validation\nr = {corr_load:.3f}', fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
os.makedirs('outputs/visualizations', exist_ok=True)
plt.savefig('outputs/visualizations/proxy_validation.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Saved: outputs/visualizations/proxy_validation.png")

# ============================================================================
# STEP 2.5: CREATE FINAL MODELING-READY DATASETS
# ============================================================================

print("\n" + "="*80)
print("STEP 2.5: CREATE FINAL FEATURE SETS")
print("="*80)

# Define final feature list for modeling
modeling_features = [
    # Core features (THE BIG ONES)
    'variance_to_expected',      # 10x lift - CRITICAL
    'duration_minutes',           # 2.07x lift
    'wait_time_minutes',          # 1.65x lift
    
    # Queue/Load features
    'queue_length_at_start',      # 1.46x lift (5k only, use proxy for 2k)
    'system_load_index_0to1',     # 1.18x lift (5k only, use proxy for 2k)
    
    # Process features
    'handoff_count_so_far',       # 0.97x lift (weak but available)
    
    # Temporal features
    'weekday',
    'hour_of_day',
    
    # Business context
    'priority',
    'sla_breached',
]

print("\nModeling features defined:")
for i, feat in enumerate(modeling_features, 1):
    print(f"  {i:2d}. {feat}")

# For 2k, map proxy features
modeling_features_2k = [
    'variance_to_expected',
    'duration_minutes',
    'wait_time_minutes',
    'concurrent_case_count',       # PROXY for queue_length
    'estimated_system_load',       # PROXY for system_load
    'handoff_count_so_far',
    'weekday',
    'hour_of_day',
    'priority',
    'sla_breached',
]

print("\n2k uses proxy features:")
print("  queue_length_at_start → concurrent_case_count")
print("  system_load_index_0to1 → estimated_system_load")

# Save processed datasets
print("\n" + "-"*80)
print("SAVING PROCESSED DATASETS")
print("-"*80)

# Save complete processed 2k
df_2k_processed.to_csv('operational-insight/data/processed/2k_processed.csv', index=False)
print(f"✓ Saved: operational-insight/data/processed/2k_processed.csv ({df_2k_processed.shape})")

# Save 5k with proxies (for validation)
df_5k_with_proxies.to_csv('operational-insight/data/processed/5k_with_proxies.csv', index=False)
print(f"✓ Saved: operational-insight/data/processed/5k_with_proxies.csv ({df_5k_with_proxies.shape})")

# ============================================================================
# MILESTONE 2 SUMMARY
# ============================================================================

summary = f"""
{'='*80}
MILESTONE 2: FEATURE ENGINEERING - COMPLETE
{'='*80}

ACHIEVEMENTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Standardized 2k features to match 5k structure
✅ Converted units (seconds → minutes)
✅ Calculated handoff_count_so_far for 2k
✅ Created hybrid expected_duration lookup (reference + data-driven)
✅ Calculated variance_to_expected (10x feature!) for 2k
✅ Engineered proxy features:
   - concurrent_case_count (queue proxy)
   - estimated_system_load (load proxy)
✅ Validated proxies on 5k:
   - Queue proxy correlation: r = {corr_queue:.3f}
   - System load proxy correlation: r = {corr_load:.3f}


ACTIVITY MISMATCH SOLUTION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Matched activities (3): Use reference table expectations
⚠️  Unmatched activities (5): Use data-driven median expectations
   - Classify Request, Enrich Data, Manual Approval, Fulfill Request, Notify Customer


DATASETS READY FOR MODELING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📁 data/processed/5k_with_proxies.csv - For training & proxy validation
📁 data/processed/2k_processed.csv - For prediction & business validation
📁 data/processed/expected_duration_lookup.csv - Reference for new data


FEATURE SET:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
10 modeling features ready:
  1. variance_to_expected (10x lift) ⭐⭐⭐
  2. duration_minutes (2.07x lift) ⭐
  3. wait_time_minutes (1.65x lift) ⭐
  4. queue_length / concurrent_case_count (1.46x lift) ⭐⭐
  5. system_load / estimated_system_load (1.18x lift)
  6. handoff_count_so_far
  7-10. weekday, hour_of_day, priority, sla_breached


NEXT MILESTONE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
→ MILESTONE 3: Model Training
  - Train Random Forest on 5k labeled data
  - Use all 10 features
  - 5-fold stratified cross-validation
  - Extract feature importances
  - Save final model

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

print(summary)

# Save summary
with open('outputs/reports/milestone2_summary.txt', 'w') as f:
    f.write(summary)

print("\n✓ Saved: outputs/reports/milestone2_summary.txt")

print("\n" + "="*80)
print("🎉 MILESTONE 2 COMPLETE!")
print("="*80)
print("\nReady for MILESTONE 3: Model Training")