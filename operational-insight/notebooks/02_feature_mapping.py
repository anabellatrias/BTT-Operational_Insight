# ============================================================================
# MILESTONE 2: FEATURE AVAILABILITY AUDIT
# ============================================================================
# Goal: Map features between 5k and 2k datasets
# Identify what we can use, what's missing, and how to handle it
# ============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)

print("="*80)
print("MILESTONE 1 - STEP 1.4: FEATURE AVAILABILITY AUDIT")
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

# Remove unnamed columns
if 'Unnamed: 0' in df_5k.columns:
    df_5k = df_5k.drop('Unnamed: 0', axis=1)
if 'Unnamed: 0' in df_2k.columns:
    df_2k = df_2k.drop('Unnamed: 0', axis=1)

print(f"\n✓ Loaded 5k dataset: {df_5k.shape}")
print(f"✓ Loaded 2k dataset: {df_2k.shape}")
print(f"✓ Loaded Activity Reference: {df_activity_ref.shape}")

# ============================================================================
# FEATURE MAPPING ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("FEATURE MAPPING: 5K → 2K")
print("="*80)

# Define feature mappings
feature_mapping = [
    # Core identifiers
    {
        '5k_column': 'case_id',
        '2k_column': 'case_id',
        'status': '✅ DIRECT MATCH',
        'transformation': 'None',
        'importance': 'Identifier'
    },
    {
        '5k_column': 'event_num_in_case',
        '2k_column': 'event_nr',
        'status': '✅ DIRECT MATCH',
        'transformation': 'Rename column',
        'importance': 'Identifier'
    },
    {
        '5k_column': 'activity_name',
        '2k_column': 'activity',
        'status': '⚠️ SIMILAR',
        'transformation': 'Different granularity - 2k has more detailed activities',
        'importance': 'HIGH - needed for reference table lookup'
    },
    
    # Target variable
    {
        '5k_column': 'is_bottleneck_event',
        '2k_column': 'MISSING',
        'status': '❌ TARGET',
        'transformation': 'This is what we predict!',
        'importance': 'TARGET VARIABLE'
    },
    
    # Duration features
    {
        '5k_column': 'duration_minutes',
        '2k_column': 'duration_sec',
        'status': '✅ CONVERTIBLE',
        'transformation': 'duration_sec / 60',
        'importance': 'HIGH - 2.07x lift for bottlenecks'
    },
    {
        '5k_column': 'wait_time_minutes',
        '2k_column': 'wait_time_sec',
        'status': '✅ CONVERTIBLE',
        'transformation': 'wait_time_sec / 60',
        'importance': 'HIGH - 1.65x lift for bottlenecks'
    },
    {
        '5k_column': 'expected_duration_minutes',
        '2k_column': 'CAN DERIVE',
        'status': '✅ CALCULABLE',
        'transformation': 'Lookup from activity_reference table',
        'importance': 'CRITICAL - needed for variance calculation'
    },
    {
        '5k_column': 'variance_to_expected',
        '2k_column': 'CAN CALCULATE',
        'status': '✅ CALCULABLE',
        'transformation': '(duration_minutes - expected) / expected',
        'importance': 'CRITICAL - 10.02x lift! Most important feature!'
    },
    
    # Queue and system features
    {
        '5k_column': 'queue_length_at_start',
        '2k_column': 'MISSING',
        'status': '❌ NEED PROXY',
        'transformation': 'Count concurrent active cases at timestamp',
        'importance': 'MEDIUM - 1.46x lift, can be proxied'
    },
    {
        '5k_column': 'system_load_index_0to1',
        '2k_column': 'MISSING',
        'status': '❌ NEED PROXY',
        'transformation': 'Rolling avg of wait_time, normalized to 0-1',
        'importance': 'LOW - 1.18x lift (not significant), less critical'
    },
    
    # Handoff features
    {
        '5k_column': 'handoff_count_so_far',
        '2k_column': 'CAN CALCULATE',
        'status': '✅ CALCULABLE',
        'transformation': 'Count resource changes per case',
        'importance': 'LOW - 0.97x (not predictive), but easy to calculate'
    },
    
    # Temporal features
    {
        '5k_column': 'weekday',
        '2k_column': 'weekday',
        'status': '✅ DIRECT MATCH',
        'transformation': 'None',
        'importance': 'MEDIUM - temporal patterns'
    },
    {
        '5k_column': 'hour_of_day',
        '2k_column': 'hour',
        'status': '✅ DIRECT MATCH',
        'transformation': 'Rename column',
        'importance': 'MEDIUM - temporal patterns'
    },
    {
        '5k_column': 'start_timestamp_utc',
        '2k_column': 'timestamp',
        'status': '⚠️ PARTIAL',
        'transformation': '2k only has one timestamp (not start/end)',
        'importance': 'MEDIUM - needed for time calculations'
    },
    
    # Business context
    {
        '5k_column': 'priority',
        '2k_column': 'priority',
        'status': '✅ DIRECT MATCH',
        'transformation': 'May need to map values (Low/Medium/High)',
        'importance': 'MEDIUM'
    },
    {
        '5k_column': 'sla_breached',
        '2k_column': 'sla_breach',
        'status': '✅ DIRECT MATCH',
        'transformation': 'Check value encoding (0/1)',
        'importance': 'MEDIUM'
    },
    
    # Additional 2k features (not in 5k)
    {
        '5k_column': 'NONE',
        '2k_column': 'cost_usd',
        'status': '➕ BONUS',
        'transformation': 'Use for business validation',
        'importance': 'HIGH - for validation (not training)'
    },
    {
        '5k_column': 'NONE',
        '2k_column': 'rework_flag',
        'status': '➕ BONUS',
        'transformation': 'Use for business validation',
        'importance': 'HIGH - for validation (not training)'
    },
]

# Create DataFrame
mapping_df = pd.DataFrame(feature_mapping)

print("\nFEATURE MAPPING TABLE:")
print("="*80)
print(mapping_df.to_string(index=False))

# ============================================================================
# FEATURE AVAILABILITY SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FEATURE AVAILABILITY SUMMARY")
print("="*80)

direct_match = len(mapping_df[mapping_df['status'].str.contains('DIRECT')])
convertible = len(mapping_df[mapping_df['status'].str.contains('CONVERTIBLE')])
calculable = len(mapping_df[mapping_df['status'].str.contains('CALCULABLE')])
need_proxy = len(mapping_df[mapping_df['status'].str.contains('PROXY')])
missing = len(mapping_df[mapping_df['status'] == '❌ TARGET'])

print(f"\n✅ Direct matches:        {direct_match}")
print(f"✅ Convertible (units):   {convertible}")
print(f"✅ Calculable (derived):  {calculable}")
print(f"❌ Need proxy features:   {need_proxy}")
print(f"❌ Target variable:       {missing}")

total_usable = direct_match + convertible + calculable
print(f"\n📊 Total usable features: {total_usable}")
print(f"⚠️  Features needing work: {need_proxy}")

# ============================================================================
# ACTIVITY MATCHING ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("ACTIVITY NAME MATCHING")
print("="*80)

print("\nActivities in 5k dataset:")
activities_5k = df_5k['activity_name'].unique()
for act in sorted(activities_5k):
    print(f"  - {act}")

print(f"\nTotal unique activities in 5k: {len(activities_5k)}")

print("\nActivities in 2k dataset:")
activities_2k = df_2k['activity'].unique()
for act in sorted(activities_2k):
    print(f"  - {act}")

print(f"\nTotal unique activities in 2k: {len(activities_2k)}")

print("\nActivities in Reference Table:")
activities_ref = df_activity_ref['activity_name'].unique()
for act in sorted(activities_ref):
    print(f"  - {act}")

print(f"\nTotal activities defined in reference: {len(activities_ref)}")

# Check matches
print("\n" + "-"*80)
print("MATCHING ANALYSIS:")
print("-"*80)

# Which 5k activities are in reference?
in_ref_5k = [act for act in activities_5k if act in activities_ref]
not_in_ref_5k = [act for act in activities_5k if act not in activities_ref]

print(f"\n5k activities IN reference table: {len(in_ref_5k)}/{len(activities_5k)}")
if not_in_ref_5k:
    print("  Missing from reference:")
    for act in not_in_ref_5k:
        print(f"    ⚠️  {act}")

# Which 2k activities are in reference?
in_ref_2k = [act for act in activities_2k if act in activities_ref]
not_in_ref_2k = [act for act in activities_2k if act not in activities_ref]

print(f"\n2k activities IN reference table: {len(in_ref_2k)}/{len(activities_2k)}")
if not_in_ref_2k:
    print("  Missing from reference:")
    for act in not_in_ref_2k:
        print(f"    ⚠️  {act}")
