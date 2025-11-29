# ============================================================================
# STREAMLIT DASHBOARD - BOTTLENECK DETECTION SYSTEM
# ============================================================================
# Interactive dashboard for exploring bottleneck predictions
# Run with: streamlit run 10_streamlit_dashboard.py
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Bottleneck Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PATH SETUP
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', '..')

# ============================================================================
# LOAD DATA AND MODEL
# ============================================================================

@st.cache_data
def load_data():
    """Load processed data and predictions"""
    file_path = os.path.join(PROJECT_ROOT, 'operational-insight', 'data', 'processed', '2k_final_predictions.csv')
    df_2k = pd.read_csv(file_path)
    return df_2k

@st.cache_resource
def load_model():
    """Load trained model - using simple random forest model"""
    # Use the simple model file that doesn't have custom classes
    file_path = os.path.join(PROJECT_ROOT, 'models', 'random_forest_model.pkl')
    package = joblib.load(file_path)
    return package

# Load data
try:
    df = load_data()
    model_package = load_model()
    model = model_package['model']
    features = model_package['features']
    
    # Add predictions if not already in dataframe
    if 'bottleneck_probability' not in df.columns:
        # Prepare features
        X = df[features].fillna(0)
        df['bottleneck_probability'] = model.predict_proba(X)[:, 1]
        df['bottleneck_prediction'] = model.predict(X)
    
except Exception as e:
    st.error(f"⚠️ Could not load data or model: {e}")
    st.info("Make sure you've run the training notebooks first!")
    st.stop()

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("🎛️ Controls")

# Filters
st.sidebar.header("📊 Filters")

# Activity filter
activities = ['All'] + sorted(df['activity_name'].unique().tolist())
selected_activity = st.sidebar.selectbox("Select Activity", activities)

# Date range (if available)
if 'case_day' in df.columns:
    try:
        df['case_day'] = pd.to_datetime(df['case_day'])
        min_date = df['case_day'].min()
        max_date = df['case_day'].max()
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    except:
        pass

# Bottleneck threshold
threshold = st.sidebar.slider(
    "Bottleneck Probability Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

# Apply filters
df_filtered = df.copy()
if selected_activity != 'All':
    df_filtered = df_filtered[df_filtered['activity_name'] == selected_activity]

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

st.markdown('<p class="main-header">🚨 Bottleneck Detection Dashboard</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================================================
# KEY METRICS ROW
# ============================================================================

col1, col2, col3, col4, col5 = st.columns(5)

total_events = len(df_filtered)
bottleneck_count = (df_filtered['bottleneck_probability'] >= threshold).sum()
bottleneck_rate = bottleneck_count / total_events * 100 if total_events > 0 else 0
avg_cost = df_filtered['cost_usd'].mean() if 'cost_usd' in df_filtered.columns else 0
avg_wait = df_filtered['wait_time_minutes'].mean() if 'wait_time_minutes' in df_filtered.columns else 0

with col1:
    st.metric(
        label="📋 Total Events",
        value=f"{total_events:,}",
        delta=None
    )

with col2:
    st.metric(
        label="🚨 Bottlenecks",
        value=f"{bottleneck_count:,}",
        delta=f"{bottleneck_rate:.1f}%"
    )

with col3:
    st.metric(
        label="💰 Avg Cost",
        value=f"${avg_cost:.2f}",
        delta=None
    )

with col4:
    st.metric(
        label="⏱️ Avg Wait Time",
        value=f"{avg_wait:.1f} min",
        delta=None
    )

with col5:
    # Calculate risk score
    risk_score = (bottleneck_rate + (avg_wait / 10)) / 2
    risk_level = "🟢 Low" if risk_score < 20 else "🟡 Medium" if risk_score < 40 else "🔴 High"
    st.metric(
        label="⚠️ Risk Level",
        value=risk_level,
        delta=f"{risk_score:.1f}"
    )

st.markdown("---")

# ============================================================================
# MAIN VISUALIZATIONS - TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Heatmap", "📊 Activity Analysis", "🎯 Predictions", "🔍 Deep Dive", "🤖 Predict New"
])

# ============================================================================
# TAB 1: BOTTLENECK HEATMAP
# ============================================================================

with tab1:
    st.header("🗺️ Bottleneck Heatmap by Activity and Time")
    
    # Create heatmap data
    if 'hour_of_day' in df_filtered.columns:
        heatmap_data = df_filtered.groupby(['activity_name', 'hour_of_day']).agg({
            'bottleneck_probability': 'mean'
        }).reset_index()
        
        heatmap_pivot = heatmap_data.pivot(
            index='activity_name',
            columns='hour_of_day',
            values='bottleneck_probability'
        )
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_pivot.values,
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            colorscale='Reds',
            text=np.round(heatmap_pivot.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Bottleneck<br>Probability")
        ))
        
        fig_heatmap.update_layout(
            title="Average Bottleneck Probability by Activity and Hour",
            xaxis_title="Hour of Day",
            yaxis_title="Activity",
            height=500,
            font=dict(size=12)
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Activity bottleneck rate
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Bottleneck Rate by Activity")
        activity_stats = df_filtered.groupby('activity_name').agg({
            'bottleneck_probability': ['mean', 'count']
        }).reset_index()
        activity_stats.columns = ['activity_name', 'avg_probability', 'count']
        activity_stats = activity_stats.sort_values('avg_probability', ascending=False)
        
        fig_activity_rate = px.bar(
            activity_stats,
            x='avg_probability',
            y='activity_name',
            orientation='h',
            title="Average Bottleneck Probability by Activity",
            labels={'avg_probability': 'Avg Probability', 'activity_name': 'Activity'},
            color='avg_probability',
            color_continuous_scale='Reds'
        )
        fig_activity_rate.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_activity_rate, use_container_width=True)
    
    with col2:
        st.subheader("🔥 Top Bottleneck Activities")
        top_bottlenecks = df_filtered[df_filtered['bottleneck_probability'] >= threshold]
        if len(top_bottlenecks) > 0:
            top_activities = top_bottlenecks['activity_name'].value_counts().head(10)
            
            fig_top = px.pie(
                values=top_activities.values,
                names=top_activities.index,
                title="Distribution of Bottleneck Events",
                hole=0.4
            )
            fig_top.update_layout(height=400)
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            st.info("No bottlenecks detected at current threshold")

# ============================================================================
# TAB 2: ACTIVITY ANALYSIS
# ============================================================================

with tab2:
    st.header("📊 Activity Duration Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Duration distribution
        if 'duration_minutes' in df_filtered.columns:
            fig_duration = px.box(
                df_filtered,
                x='activity_name',
                y='duration_minutes',
                color='activity_name',
                title="Duration Distribution by Activity",
                labels={'duration_minutes': 'Duration (minutes)', 'activity_name': 'Activity'}
            )
            fig_duration.update_layout(
                showlegend=False,
                height=400,
                xaxis={'tickangle': -45}
            )
            st.plotly_chart(fig_duration, use_container_width=True)
    
    with col2:
        # Wait time distribution
        if 'wait_time_minutes' in df_filtered.columns:
            fig_wait = px.box(
                df_filtered,
                x='activity_name',
                y='wait_time_minutes',
                color='activity_name',
                title="Wait Time Distribution by Activity",
                labels={'wait_time_minutes': 'Wait Time (minutes)', 'activity_name': 'Activity'}
            )
            fig_wait.update_layout(
                showlegend=False,
                height=400,
                xaxis={'tickangle': -45}
            )
            st.plotly_chart(fig_wait, use_container_width=True)
    
    # Detailed activity table
    st.subheader("📋 Activity Statistics")
    
    agg_dict = {'case_id': 'count', 'bottleneck_probability': 'mean'}
    if 'duration_minutes' in df_filtered.columns:
        agg_dict['duration_minutes'] = ['mean', 'median', 'std']
    if 'wait_time_minutes' in df_filtered.columns:
        agg_dict['wait_time_minutes'] = ['mean', 'median']
    if 'cost_usd' in df_filtered.columns:
        agg_dict['cost_usd'] = 'mean'
    
    activity_detailed = df_filtered.groupby('activity_name').agg(agg_dict).round(2)
    
    # Flatten column names
    activity_detailed.columns = ['_'.join(col).strip('_') for col in activity_detailed.columns.values]
    activity_detailed = activity_detailed.reset_index()
    
    st.dataframe(activity_detailed, use_container_width=True, hide_index=True)

# ============================================================================
# TAB 3: PREDICTION SUMMARY
# ============================================================================

with tab3:
    st.header("🎯 Prediction Summary")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        # Probability distribution
        fig_prob_dist = px.histogram(
            df_filtered,
            x='bottleneck_probability',
            nbins=50,
            title="Bottleneck Probability Distribution",
            labels={'bottleneck_probability': 'Probability'},
            color_discrete_sequence=['steelblue']
        )
        fig_prob_dist.add_vline(x=threshold, line_dash="dash", line_color="red",
                               annotation_text=f"Threshold: {threshold}")
        fig_prob_dist.update_layout(height=400)
        st.plotly_chart(fig_prob_dist, use_container_width=True)
    
    with col2:
        # Confidence levels
        df_filtered['confidence_level'] = pd.cut(
            df_filtered['bottleneck_probability'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        confidence_counts = df_filtered['confidence_level'].value_counts()
        
        fig_confidence = px.pie(
            values=confidence_counts.values,
            names=confidence_counts.index,
            title="Risk Level Distribution",
            color=confidence_counts.index,
            color_discrete_map={
                'Low Risk': 'green',
                'Medium Risk': 'yellow',
                'High Risk': 'red'
            },
            hole=0.4
        )
        fig_confidence.update_layout(height=400)
        st.plotly_chart(fig_confidence, use_container_width=True)
    
    with col3:
        st.subheader("📊 Statistics")
        st.metric("Mean Probability", f"{df_filtered['bottleneck_probability'].mean():.3f}")
        st.metric("Median Probability", f"{df_filtered['bottleneck_probability'].median():.3f}")
        st.metric("Std Dev", f"{df_filtered['bottleneck_probability'].std():.3f}")
        st.metric("High Risk (>0.7)", f"{(df_filtered['bottleneck_probability'] > 0.7).sum()}")
    
    # Top predicted bottlenecks
    st.subheader("🚨 Top 20 Predicted Bottlenecks")
    
    display_cols = ['case_id', 'activity_name', 'bottleneck_probability']
    if 'duration_minutes' in df_filtered.columns:
        display_cols.append('duration_minutes')
    if 'wait_time_minutes' in df_filtered.columns:
        display_cols.append('wait_time_minutes')
    if 'cost_usd' in df_filtered.columns:
        display_cols.append('cost_usd')
    
    top_predictions = df_filtered.nlargest(20, 'bottleneck_probability')[display_cols].round(3)
    
    st.dataframe(
        top_predictions.style.background_gradient(subset=['bottleneck_probability'], cmap='Reds'),
        use_container_width=True,
        hide_index=True
    )

# ============================================================================
# TAB 4: DEEP DIVE
# ============================================================================

with tab4:
    st.header("🔍 Deep Dive Analysis")
    
    # Scatter plots
    col1, col2 = st.columns(2)
    
    with col1:
        if 'wait_time_minutes' in df_filtered.columns and 'duration_minutes' in df_filtered.columns:
            fig_scatter1 = px.scatter(
                df_filtered.sample(min(500, len(df_filtered))),  # Sample for performance
                x='wait_time_minutes',
                y='duration_minutes',
                color='bottleneck_probability',
                title="Wait Time vs Duration (colored by bottleneck probability)",
                labels={'wait_time_minutes': 'Wait Time (min)', 'duration_minutes': 'Duration (min)'},
                color_continuous_scale='Reds',
                opacity=0.6
            )
            fig_scatter1.update_layout(height=400)
            st.plotly_chart(fig_scatter1, use_container_width=True)
    
    with col2:
        if 'cost_usd' in df_filtered.columns:
            fig_scatter2 = px.scatter(
                df_filtered.sample(min(500, len(df_filtered))),
                x='cost_usd',
                y='bottleneck_probability',
                color='activity_name',
                title="Cost vs Bottleneck Probability",
                labels={'cost_usd': 'Cost ($)', 'bottleneck_probability': 'Bottleneck Prob'},
                opacity=0.6
            )
            fig_scatter2.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_scatter2, use_container_width=True)
    
    # Case analysis
    st.subheader("📦 Case-Level Analysis")
    
    agg_dict = {
        'bottleneck_probability': 'max',
        'activity_name': 'count'
    }
    if 'duration_minutes' in df_filtered.columns:
        agg_dict['duration_minutes'] = 'sum'
    if 'wait_time_minutes' in df_filtered.columns:
        agg_dict['wait_time_minutes'] = 'sum'
    if 'cost_usd' in df_filtered.columns:
        agg_dict['cost_usd'] = 'sum'
    
    case_stats = df_filtered.groupby('case_id').agg(agg_dict).reset_index()
    
    # Rename columns
    col_rename = {'activity_name': 'num_activities', 'bottleneck_probability': 'max_bottleneck_prob'}
    if 'duration_minutes' in case_stats.columns:
        col_rename['duration_minutes'] = 'total_duration'
    if 'wait_time_minutes' in case_stats.columns:
        col_rename['wait_time_minutes'] = 'total_wait'
    if 'cost_usd' in case_stats.columns:
        col_rename['cost_usd'] = 'total_cost'
    
    case_stats = case_stats.rename(columns=col_rename)
    case_stats = case_stats.sort_values('max_bottleneck_prob', ascending=False).head(20)
    
    fig_cases = px.bar(
        case_stats,
        x='case_id',
        y='max_bottleneck_prob',
        title="Top 20 Cases by Max Bottleneck Probability",
        labels={'max_bottleneck_prob': 'Max Bottleneck Prob', 'case_id': 'Case ID'},
        color='max_bottleneck_prob',
        color_continuous_scale='Reds'
    )
    fig_cases.update_layout(height=400, xaxis={'tickangle': -45})
    st.plotly_chart(fig_cases, use_container_width=True)

# ============================================================================
# TAB 5: PREDICT NEW EVENT
# ============================================================================

with tab5:
    st.header("🤖 Predict New Event")
    st.write("Enter event details to get a bottleneck prediction:")
    
    col1, col2 = st.columns(2)
    
    # Get feature names and create inputs dynamically
    with col1:
        feature_inputs = {}
        
        if 'variance_to_expected' in features:
            feature_inputs['variance_to_expected'] = st.number_input(
                "Variance to Expected",
                min_value=-1.0,
                max_value=10.0,
                value=0.0,
                step=0.1,
                help="How much longer/shorter than expected (0 = on time, 0.5 = 50% over)"
            )
        
        if 'duration_minutes' in features:
            feature_inputs['duration_minutes'] = st.number_input(
                "Duration (minutes)",
                min_value=0.0,
                max_value=500.0,
                value=10.0,
                step=1.0
            )
        
        if 'wait_time_minutes' in features:
            feature_inputs['wait_time_minutes'] = st.number_input(
                "Wait Time (minutes)",
                min_value=0.0,
                max_value=500.0,
                value=5.0,
                step=1.0
            )
    
    with col2:
        if 'handoff_count_so_far' in features:
            feature_inputs['handoff_count_so_far'] = st.number_input(
                "Handoff Count",
                min_value=0,
                max_value=20,
                value=1,
                step=1
            )
        
        if 'hour_of_day' in features:
            feature_inputs['hour_of_day'] = st.number_input(
                "Hour of Day",
                min_value=0,
                max_value=23,
                value=9,
                step=1
            )
        
        if 'sla_breached' in features:
            feature_inputs['sla_breached'] = st.selectbox(
                "SLA Breached",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes"
            )
    
    if st.button("🎯 Predict Bottleneck", type="primary"):
        # Prepare input - make sure all features are present
        input_data = {}
        for feat in features:
            if feat in feature_inputs:
                input_data[feat] = feature_inputs[feat]
            else:
                input_data[feat] = 0  # Default value for missing features
        
        input_features = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_features)[0]
        probability = model.predict_proba(input_features)[0][1]
        
        # Display results
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.error("🚨 **BOTTLENECK DETECTED**")
            else:
                st.success("✅ **NORMAL EVENT**")
        
        with col2:
            st.metric(
                "Bottleneck Probability",
                f"{probability:.1%}",
                delta=f"{(probability - 0.5)*100:+.1f}%" if probability != 0.5 else None
            )
        
        with col3:
            risk = "🔴 High" if probability > 0.7 else "🟡 Medium" if probability > 0.3 else "🟢 Low"
            st.metric("Risk Level", risk)
        
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Bottleneck Probability"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Feature values
        st.subheader("📊 Input Values")
        input_df = pd.DataFrame([feature_inputs])
        st.dataframe(input_df, use_container_width=True, hide_index=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Bottleneck Detection System v1.0</strong></p>
    <p>Built with Streamlit • Powered by Random Forest ML</p>
    <p>Model F1 Score: 0.989 | Features: {len(features)} | Training Size: 2,000 events</p>
</div>
""", unsafe_allow_html=True)