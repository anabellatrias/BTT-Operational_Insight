# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Use repo root (one level up from notebooks) for consistent data paths
repo_dir = Path(__file__).resolve().parents[1]
data_path = repo_dir / "data" / "raw" 

# Display settings for better output
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)

print("✓ Libraries imported successfully")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")

# ============================================================================
# STEP 1: LOAD AND INSPECT DATASETS
# ============================================================================

print("="*80)
print("LOADING DATASETS")
print("="*80)

# Load datasets
try:
    df_5k = pd.read_csv(data_path / "5kp.csv")
    print("✓ Loaded 5k dataset")
    
    df_2k = pd.read_csv(data_path / "2k.csv")
    print("✓ Loaded 2k dataset")
    
    df_activity_ref = pd.read_csv(data_path / "Activity_Reference.csv")
    print("✓ Loaded Activity Reference")
    
    df_dictionary = pd.read_csv(data_path / "Event_Log_Dictionary.csv")
    print("✓ Loaded Event Log Dictionary")
    
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please check your file paths!")
    raise

print("\n" + "="*80)
print("INITIAL DATASET SHAPES")
print("="*80)
print(f"5k dataset:           {df_5k.shape[0]:,} rows × {df_5k.shape[1]} columns")
print(f"2k dataset:           {df_2k.shape[0]:,} rows × {df_2k.shape[1]} columns")
print(f"Activity Reference:   {df_activity_ref.shape[0]:,} rows × {df_activity_ref.shape[1]} columns")
print(f"Data Dictionary:      {df_dictionary.shape[0]:,} rows × {df_dictionary.shape[1]} columns")

# ============================================================================
# QUICK PREVIEW OF EACH DATASET
# ============================================================================

print("\n" + "="*80)
print("5K DATASET - First 3 Rows")
print("="*80)
print(df_5k.head(3))

print("\n" + "="*80)
print("2K DATASET - First 3 Rows")
print("="*80)
print(df_2k.head(3))

print("\n" + "="*80)
print("ACTIVITY REFERENCE TABLE")
print("="*80)
print(df_activity_ref)

print("\n" + "="*80)
print("5K COLUMN NAMES")
print("="*80)
print(df_5k.columns.tolist())

print("\n" + "="*80)
print("2K COLUMN NAMES")
print("="*80)
print(df_2k.columns.tolist())

# ============================================================================
# DATA QUALITY ASSESSMENT
# ============================================================================

def assess_data_quality(df, dataset_name):
    """
    Comprehensive data quality check
    """
    print(f"\n{'='*80}")
    print(f"DATA QUALITY ASSESSMENT: {dataset_name}")
    print(f"{'='*80}")
    
    # Basic info
    print(f"\nDataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Missing values
    print(f"\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing_Count': missing.values,
        'Missing_Percentage': missing_pct.values
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df.to_string(index=False))
    else:
        print("✓ No missing values found!")
    
    # Duplicates
    print(f"\n--- Duplicate Rows ---")
    duplicates = df.duplicated().sum()
    print(f"Total duplicate rows: {duplicates}")
    if duplicates > 0:
        print("⚠️ Warning: Duplicates found!")
    else:
        print("✓ No duplicate rows")
    
    # Data types
    print(f"\n--- Data Types ---")
    print(df.dtypes.value_counts())
    
    return missing_df

# Assess each dataset
missing_5k = assess_data_quality(df_5k, "5K LABELED DATASET")
missing_2k = assess_data_quality(df_2k, "2K OPERATIONAL DATASET")
missing_ref = assess_data_quality(df_activity_ref, "ACTIVITY REFERENCE")

# ============================================================================
# DATA TYPE CORRECTIONS
# ============================================================================

print("\n" + "="*80)
print("FIXING DATA TYPES")
print("="*80)

# Fix timestamps in 5k dataset
if 'start_timestamp_utc' in df_5k.columns:
    df_5k['start_timestamp_utc'] = pd.to_datetime(df_5k['start_timestamp_utc'])
    df_5k['end_timestamp_utc'] = pd.to_datetime(df_5k['end_timestamp_utc'])
    print("✓ Converted 5k timestamps to datetime")
else:
    print("⚠️ Timestamp columns not found in 5k dataset")

# Fix timestamps in 2k dataset
if 'timestamp' in df_2k.columns:
    df_2k['timestamp'] = pd.to_datetime(df_2k['timestamp'])
    print("✓ Converted 2k timestamps to datetime")
else:
    print("⚠️ Timestamp column not found in 2k dataset")

# Check if unnamed columns exist (from CSV index)
if 'Unnamed: 0' in df_5k.columns:
    df_5k = df_5k.drop('Unnamed: 0', axis=1)
    print("✓ Removed unnecessary index column from 5k")

if 'Unnamed: 0' in df_2k.columns:
    df_2k = df_2k.drop('Unnamed: 0', axis=1)
    print("✓ Removed unnecessary index column from 2k")

print("\n✓ Data types corrected")

# ============================================================================
# CLASS DISTRIBUTION ANALYSIS (CRITICAL)
# ============================================================================

print("\n" + "="*80)
print("CLASS DISTRIBUTION ANALYSIS")
print("="*80)

# Check if target variable exists
if 'is_bottleneck_event' not in df_5k.columns:
    print("ERROR: 'is_bottleneck_event' column not found in 5k dataset!")
    print("Available columns:", df_5k.columns.tolist())
else:
    # Count bottlenecks
    class_counts = df_5k['is_bottleneck_event'].value_counts().sort_index()
    class_pct = df_5k['is_bottleneck_event'].value_counts(normalize=True).sort_index() * 100
    
    print("\n--- Class Distribution ---")
    print(f"Normal events (0):     {class_counts[0]:3d} ({class_pct[0]:.1f}%)")
    print(f"Bottleneck events (1): {class_counts[1]:3d} ({class_pct[1]:.1f}%)")
    print(f"Total events:          {len(df_5k):3d}")
    
    bottleneck_rate = class_pct[1]
    
    # Assessment
    print("\n--- Class Balance Assessment ---")
    if bottleneck_rate < 5:
        print(f"⚠️ SEVERE IMBALANCE: Only {bottleneck_rate:.1f}% bottlenecks")
        print("   → Consider: Rule-based approach or anomaly detection")
    elif bottleneck_rate < 20:
        print(f"⚠️ MODERATE IMBALANCE: {bottleneck_rate:.1f}% bottlenecks")
        print("   → Must use: class_weight='balanced' in models")
    elif bottleneck_rate > 80:
        print(f"⚠️ INVERTED IMBALANCE: {bottleneck_rate:.1f}% bottlenecks")
        print("   → Too many positives, check data quality")
    else:
        print(f"✓ ACCEPTABLE BALANCE: {bottleneck_rate:.1f}% bottlenecks")
        print("   → Machine learning is feasible")
    
    # Visualization for Class Imbalance on 5k dataset
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    axes[0].bar(['Normal', 'Bottleneck'], class_counts.values, 
                color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Class Distribution (Absolute Counts)', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 1, str(v), ha='center', fontweight='bold', fontsize=12)
    
    # Pie chart
    colors = ['#2ecc71', '#e74c3c']
    axes[1].pie(class_counts.values, labels=['Normal', 'Bottleneck'], autopct='%1.1f%%',
                colors=colors, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[1].set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    import os
    os.makedirs('outputs/visualizations', exist_ok=True)
    
    plt.savefig('outputs/visualizations/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Visualization saved to: outputs/visualizations/class_distribution.png")

# ============================================================================
# SAVE CHECKPOINT REPORT
# ============================================================================

report = f"""
MILESTONE 1 - DATA UNDERSTANDING CHECKPOINT
{'='*80}

DATASETS LOADED:
- 5k dataset: {df_5k.shape[0]} rows × {df_5k.shape[1]} columns
- 2k dataset: {df_2k.shape[0]} rows × {df_2k.shape[1]} columns
- Activity Reference: {df_activity_ref.shape[0]} activities defined

CLASS DISTRIBUTION (5k):
- Normal events: {class_counts[0]} ({class_pct[0]:.1f}%)
- Bottleneck events: {class_counts[1]} ({class_pct[1]:.1f}%)
- Assessment: {"VIABLE FOR ML" if 20 <= bottleneck_rate <= 80 else "CONSIDER ALTERNATIVES"}

DATA QUALITY:
- Missing values in 5k: 0 columns affected
- Missing values in 2k: 1 column (outcome - not needed)
- Duplicates: None detected

NEXT STEPS:
- Continue to Exploratory Data Analysis
- Compare bottleneck vs normal event characteristics

Generated: {pd.Timestamp.now()}
"""

# Save report
with open('outputs/reports/milestone1_checkpoint.txt', 'w') as f:
    f.write(report)

print(report)
print("\n✓ Checkpoint report saved to: outputs/reports/milestone1_checkpoint.txt")

# ============================================================================
# EXPLORATORY DATA ANALYSIS - BOTTLENECK PATTERNS
# ============================================================================

print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS")
print("="*80)

# Separate bottleneck vs normal events
bottlenecks = df_5k[df_5k['is_bottleneck_event'] == 1].copy()
normal = df_5k[df_5k['is_bottleneck_event'] == 0].copy()

print(f"\nAnalyzing {len(bottlenecks)} bottleneck events vs {len(normal)} normal events")

# ============================================================================
# KEY METRICS COMPARISON
# ============================================================================

print("\n" + "-"*80)
print("KEY METRICS COMPARISON: Bottleneck vs Normal")
print("-"*80)

# Define metrics to compare
metrics = {
    'duration_minutes': 'Duration (minutes)',
    'wait_time_minutes': 'Wait Time (minutes)',
    'queue_length_at_start': 'Queue Length',
    'system_load_index_0to1': 'System Load (0-1)',
    'variance_to_expected': 'Variance to Expected',
    'handoff_count_so_far': 'Handoff Count'
}

comparison_results = []

for col, label in metrics.items():
    if col in df_5k.columns:
        # Calculate statistics
        bottleneck_mean = bottlenecks[col].mean()
        normal_mean = normal[col].mean()
        
        # Calculate lift (how much higher bottlenecks are)
        if normal_mean != 0:
            lift = bottleneck_mean / normal_mean
        else:
            lift = np.inf
        
        # Statistical test (t-test)
        t_stat, p_value = stats.ttest_ind(
            bottlenecks[col].dropna(), 
            normal[col].dropna()
        )
        
        # Is it significant?
        significant = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        
        comparison_results.append({
            'Metric': label,
            'Bottleneck_Avg': bottleneck_mean,
            'Normal_Avg': normal_mean,
            'Lift': lift,
            'P_Value': p_value,
            'Significant': significant
        })
        
        print(f"\n{label}:")
        print(f"  Bottleneck avg: {bottleneck_mean:8.2f}")
        print(f"  Normal avg:     {normal_mean:8.2f}")
        print(f"  Lift:           {lift:8.2f}x {significant}")
        print(f"  P-value:        {p_value:.4f}")

# Create comparison dataframe
comparison_df = pd.DataFrame(comparison_results)

print("\n" + "="*80)
print("SUMMARY: Top Bottleneck Indicators")
print("="*80)
print("\nMetrics ranked by lift (higher = stronger bottleneck signal):")
print(comparison_df.sort_values('Lift', ascending=False)[
    ['Metric', 'Lift', 'Significant']
].to_string(index=False))

# ============================================================================
# VISUALIZATIONS: Feature Distributions
# ============================================================================

print("\n" + "="*80)
print("GENERATING COMPARISON VISUALIZATIONS")
print("="*80)

# Create figure with subplots
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flatten()

plot_idx = 0
for col, label in metrics.items():
    if col in df_5k.columns and plot_idx < 6:
        ax = axes[plot_idx]
        
        # Violin plot
        data_to_plot = [
            normal[col].dropna(),
            bottlenecks[col].dropna()
        ]
        
        parts = ax.violinplot(
            data_to_plot,
            positions=[0, 1],
            showmeans=True,
            showmedians=True
        )
        
        # Color the violins
        for pc, color in zip(parts['bodies'], ['#2ecc71', '#e74c3c']):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        # Add box plots on top
        bp = ax.boxplot(
            data_to_plot,
            positions=[0, 1],
            widths=0.3,
            patch_artist=True,
            showfliers=False
        )
        
        for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c']):
            patch.set_facecolor(color)
            patch.set_alpha(0.3)
        
        # Labels
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Normal', 'Bottleneck'], fontweight='bold')
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label} Distribution', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add lift annotation
        lift = comparison_df[comparison_df['Metric'] == label]['Lift'].values[0]
        sig = comparison_df[comparison_df['Metric'] == label]['Significant'].values[0]
        ax.text(0.5, 0.95, f'Lift: {lift:.2f}x {sig}', 
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10, fontweight='bold')
        
        plot_idx += 1

plt.tight_layout()
plt.savefig('outputs/visualizations/feature_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Saved: outputs/visualizations/feature_distributions.png")

# ============================================================================
# ACTIVITY-LEVEL ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("ACTIVITY-LEVEL BOTTLENECK ANALYSIS")
print("="*80)

activity_analysis = df_5k.groupby('activity_name').agg({
    'is_bottleneck_event': ['sum', 'count', 'mean']
}).round(3)

activity_analysis.columns = ['Bottlenecks', 'Total_Events', 'Bottleneck_Rate']
activity_analysis = activity_analysis.sort_values('Bottleneck_Rate', ascending=False)

print("\nBottleneck Rate by Activity:")
print(activity_analysis)

# Visualize
fig, ax = plt.subplots(figsize=(12, 6))
activity_analysis['Bottleneck_Rate'].plot(
    kind='barh', 
    ax=ax, 
    color=['#e74c3c' if x > 0.5 else '#f39c12' if x > 0.3 else '#2ecc71' 
           for x in activity_analysis['Bottleneck_Rate']]
)
ax.set_xlabel('Bottleneck Rate', fontsize=12, fontweight='bold')
ax.set_title('Which Activities Are Most Likely to Be Bottlenecks?', 
             fontsize=14, fontweight='bold')
ax.axvline(x=0.38, color='black', linestyle='--', linewidth=2, label='Overall Rate (38%)')
ax.legend()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/visualizations/activity_bottleneck_rates.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Saved: outputs/visualizations/activity_bottleneck_rates.png")

# ============================================================================
# TEMPORAL PATTERNS
# ============================================================================

print("\n" + "="*80)
print("TEMPORAL BOTTLENECK PATTERNS")
print("="*80)

# By hour of day
hour_analysis = df_5k.groupby('hour_of_day')['is_bottleneck_event'].agg(['sum', 'count', 'mean'])
hour_analysis.columns = ['Bottlenecks', 'Total', 'Rate']

print("\nBottleneck Rate by Hour of Day:")
print(hour_analysis[hour_analysis['Total'] > 0].sort_values('Rate', ascending=False).head(10))

# By weekday
weekday_analysis = df_5k.groupby('weekday')['is_bottleneck_event'].agg(['sum', 'count', 'mean'])
weekday_analysis.columns = ['Bottlenecks', 'Total', 'Rate']

print("\nBottleneck Rate by Weekday:")
print(weekday_analysis.sort_values('Rate', ascending=False))

# ============================================================================
# CORRELATION HEATMAP
# ============================================================================

print("\n" + "="*80)
print("FEATURE CORRELATION ANALYSIS")
print("="*80)

# Select numeric features
numeric_cols = [
    'duration_minutes', 'wait_time_minutes', 'queue_length_at_start',
    'system_load_index_0to1', 'variance_to_expected', 'handoff_count_so_far',
    'hour_of_day', 'is_bottleneck_event'
]

correlation_matrix = df_5k[numeric_cols].corr()

# Plot heatmap
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    fmt='.2f', 
    cmap='RdYlGn_r',
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={"shrink": 0.8},
    ax=ax
)
ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('outputs/visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Saved: outputs/visualizations/correlation_heatmap.png")

# Print correlations with target
print("\nCorrelation with Bottleneck Target:")
target_corr = correlation_matrix['is_bottleneck_event'].sort_values(ascending=False)
print(target_corr.to_string())

# ============================================================================
# KEY INSIGHTS SUMMARY
# ============================================================================

print("\n" + "="*80)
print("KEY INSIGHTS FROM EDA")
print("="*80)

# Find top 3 predictive features
top_features = comparison_df.nlargest(3, 'Lift')

print("\n✅ TOP 3 BOTTLENECK INDICATORS:")
for idx, row in top_features.iterrows():
    print(f"\n{idx+1}. {row['Metric']}")
    print(f"   - Bottlenecks are {row['Lift']:.2f}x higher than normal {row['Significant']}")
    print(f"   - Statistical significance: p={row['P_Value']:.4f}")

# Most problematic activities
top_activity = activity_analysis.index[0]
top_activity_rate = activity_analysis.iloc[0]['Bottleneck_Rate']

print(f"\n✅ MOST PROBLEMATIC ACTIVITY:")
print(f"   - '{top_activity}' has {top_activity_rate:.1%} bottleneck rate")
print(f"   - This is {top_activity_rate/0.38:.2f}x higher than overall rate")
