"""Streamlit demo application for Employee Satisfaction Analysis.

This application provides an interactive interface for exploring employee
satisfaction data, training models, and generating insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import os
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data import EmployeeDataGenerator, DataProcessor
from src.features import FeatureEngineer
from src.models import SentimentAnalyzer, SatisfactionPredictor, FairnessAnalyzer
from src.eval import ModelEvaluator, VisualizationGenerator, ReportGenerator

# Page configuration
st.set_page_config(
    page_title="Employee Satisfaction Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .insight-card {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error(f"Configuration file not found: {config_path}")
        return {}

def initialize_session_state():
    """Initialize session state variables."""
    if 'data_generated' not in st.session_state:
        st.session_state.data_generated = False
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Employee Satisfaction Analysis</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ IMPORTANT DISCLAIMER:</strong><br>
        This is an experimental research and educational tool. This application is NOT intended for 
        automated decision-making without human review. All outputs should be validated by qualified 
        HR professionals and used only as supplementary information for human decision-making processes.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Data Explorer", "🤖 Model Training", "📈 Evaluation", "🔍 Insights", "⚙️ Settings"]
    )
    
    # Load configurations
    config_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'configs')
    sentiment_config = load_config(os.path.join(config_dir, 'sentiment_config.yaml'))
    satisfaction_config = load_config(os.path.join(config_dir, 'satisfaction_config.yaml'))
    eval_config = load_config(os.path.join(config_dir, 'eval_config.yaml'))
    
    # Route to appropriate page
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Explorer":
        show_data_explorer_page()
    elif page == "🤖 Model Training":
        show_model_training_page(sentiment_config, satisfaction_config)
    elif page == "📈 Evaluation":
        show_evaluation_page(eval_config)
    elif page == "🔍 Insights":
        show_insights_page()
    elif page == "⚙️ Settings":
        show_settings_page()

def show_home_page():
    """Display the home page."""
    st.markdown("## Welcome to Employee Satisfaction Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Purpose
        This tool helps organizations understand workforce morale and engagement through:
        - Advanced sentiment analysis of employee feedback
        - Machine learning-based satisfaction prediction
        - Fairness-aware analytics
        - Actionable business insights
        """)
    
    with col2:
        st.markdown("""
        ### 🔧 Features
        - **Sentiment Analysis**: Multi-model ensemble approach
        - **Satisfaction Prediction**: ML models with feature engineering
        - **Fairness Analysis**: Bias detection and mitigation
        - **Business Metrics**: Department comparisons and trends
        - **Interactive Visualizations**: Real-time data exploration
        """)
    
    with col3:
        st.markdown("""
        ### 📋 Quick Start
        1. **Generate Data**: Create synthetic employee data
        2. **Explore Data**: Visualize satisfaction patterns
        3. **Train Model**: Build satisfaction prediction model
        4. **Evaluate**: Assess model performance and fairness
        5. **Get Insights**: Generate actionable recommendations
        """)
    
    # Data generation section
    st.markdown("## 🚀 Quick Start")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("Generate synthetic employee satisfaction data to get started:")
    
    with col2:
        if st.button("🎲 Generate Sample Data", type="primary"):
            with st.spinner("Generating employee data..."):
                generator = EmployeeDataGenerator()
                data = generator.generate_employee_data(n_samples=1000)
                st.session_state.data = data
                st.session_state.data_generated = True
                st.success(f"Generated {len(data)} employee records!")
                st.rerun()
    
    # Show data summary if available
    if st.session_state.data_generated and st.session_state.data is not None:
        st.markdown("### 📊 Data Summary")
        
        data = st.session_state.data
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Employees", len(data))
        
        with col2:
            avg_satisfaction = data['satisfaction_score'].mean()
            st.metric("Avg Satisfaction", f"{avg_satisfaction:.2f}")
        
        with col3:
            positive_rate = (data['satisfaction_level'] == 'positive').mean()
            st.metric("Positive Rate", f"{positive_rate:.1%}")
        
        with col4:
            departments = data['department'].nunique()
            st.metric("Departments", departments)

def show_data_explorer_page():
    """Display the data explorer page."""
    st.markdown("## 📊 Data Explorer")
    
    if not st.session_state.data_generated:
        st.warning("Please generate data first from the Home page.")
        return
    
    data = st.session_state.data
    
    # Data overview
    st.markdown("### Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Sample Data")
        st.dataframe(data.head(10))
    
    with col2:
        st.markdown("#### Data Info")
        st.write(f"**Shape:** {data.shape}")
        st.write(f"**Columns:** {list(data.columns)}")
        st.write(f"**Date Range:** {data['response_date'].min().strftime('%Y-%m-%d')} to {data['response_date'].max().strftime('%Y-%m-%d')}")
    
    # Visualizations
    st.markdown("### 📈 Visualizations")
    
    # Satisfaction distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(data, x='satisfaction_score', nbins=20, title='Satisfaction Score Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        satisfaction_counts = data['satisfaction_level'].value_counts()
        fig = px.pie(values=satisfaction_counts.values, names=satisfaction_counts.index, 
                     title='Satisfaction Level Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # Department analysis
    st.markdown("#### Department Analysis")
    
    dept_stats = data.groupby('department').agg({
        'satisfaction_score': ['mean', 'count'],
        'satisfaction_level': lambda x: (x == 'positive').mean()
    }).round(3)
    
    dept_stats.columns = ['avg_satisfaction', 'count', 'positive_rate']
    dept_stats = dept_stats.sort_values('avg_satisfaction', ascending=True)
    
    fig = px.bar(dept_stats.reset_index(), x='avg_satisfaction', y='department', 
                 orientation='h', title='Average Satisfaction by Department')
    st.plotly_chart(fig, use_container_width=True)
    
    # Role level analysis
    st.markdown("#### Role Level Analysis")
    
    role_stats = data.groupby('role_level').agg({
        'satisfaction_score': ['mean', 'count'],
        'satisfaction_level': lambda x: (x == 'positive').mean()
    }).round(3)
    
    role_stats.columns = ['avg_satisfaction', 'count', 'positive_rate']
    role_stats = role_stats.sort_values('avg_satisfaction', ascending=True)
    
    fig = px.bar(role_stats.reset_index(), x='avg_satisfaction', y='role_level', 
                 orientation='h', title='Average Satisfaction by Role Level')
    st.plotly_chart(fig, use_container_width=True)
    
    # Time series analysis
    st.markdown("#### Time Series Analysis")
    
    data_with_date = data.copy()
    data_with_date['month'] = data_with_date['response_date'].dt.to_period('M')
    monthly_stats = data_with_date.groupby('month')['satisfaction_score'].mean().reset_index()
    monthly_stats['month'] = monthly_stats['month'].astype(str)
    
    fig = px.line(monthly_stats, x='month', y='satisfaction_score', 
                  title='Satisfaction Trends Over Time')
    st.plotly_chart(fig, use_container_width=True)

def show_model_training_page(sentiment_config, satisfaction_config):
    """Display the model training page."""
    st.markdown("## 🤖 Model Training")
    
    if not st.session_state.data_generated:
        st.warning("Please generate data first from the Home page.")
        return
    
    data = st.session_state.data
    
    # Model configuration
    st.markdown("### Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Model Type",
            ["gradient_boosting", "random_forest"],
            help="Choose the machine learning model type"
        )
        
        n_estimators = st.slider("Number of Estimators", 50, 200, 100)
        max_depth = st.slider("Max Depth", 3, 15, 6)
    
    with col2:
        learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, step=0.01)
        test_size = st.slider("Test Size", 0.1, 0.3, 0.2, step=0.05)
        
        # Update config
        satisfaction_config['model']['type'] = model_type
        satisfaction_config['gradient_boosting']['n_estimators'] = n_estimators
        satisfaction_config['gradient_boosting']['max_depth'] = max_depth
        satisfaction_config['gradient_boosting']['learning_rate'] = learning_rate
    
    # Training section
    st.markdown("### Training")
    
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            try:
                # Feature engineering
                st.write("🔧 Engineering features...")
                feature_engineer = FeatureEngineer()
                data_processed = feature_engineer.fit_transform(data)
                
                # Prepare data
                X = data_processed.drop(['satisfaction_score', 'satisfaction_level', 'employee_id'], axis=1, errors='ignore')
                y = data_processed['satisfaction_score']
                
                # Train-test split
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=None
                )
                
                # Train model
                st.write("🤖 Training satisfaction predictor...")
                model = SatisfactionPredictor(satisfaction_config)
                model.fit(X_train, y_train)
                
                # Store in session state
                st.session_state.model = model
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.model_trained = True
                
                st.success("Model trained successfully!")
                
                # Show training metrics
                train_score = model.model.score(X_train, y_train)
                st.metric("Training R² Score", f"{train_score:.4f}")
                
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
    
    # Model information
    if st.session_state.model_trained:
        st.markdown("### Model Information")
        
        model = st.session_state.model
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Type:**", model.config['model']['type'])
            st.write("**Features Used:**", len(model.feature_names))
            st.write("**Feature Names:**", model.feature_names[:10])  # Show first 10
        
        with col2:
            # Feature importance
            if hasattr(model, 'get_feature_importance'):
                importance_df = model.get_feature_importance()
                top_features = importance_df.head(10)
                
                fig = px.bar(top_features, x='importance', y='feature', 
                           orientation='h', title='Top 10 Feature Importance')
                st.plotly_chart(fig, use_container_width=True)

def show_evaluation_page(eval_config):
    """Display the evaluation page."""
    st.markdown("## 📈 Model Evaluation")
    
    if not st.session_state.model_trained:
        st.warning("Please train a model first from the Model Training page.")
        return
    
    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    
    # Evaluation metrics
    st.markdown("### Performance Metrics")
    
    if st.button("📊 Evaluate Model", type="primary"):
        with st.spinner("Evaluating model..."):
            try:
                # Generate predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                # Binary classification metrics
                y_test_binary = (y_test >= 3.0).astype(int)
                y_pred_binary = (y_pred >= 3.0).astype(int)
                accuracy = accuracy_score(y_test_binary, y_pred_binary)
                
                # Store results
                evaluation_results = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'true_values': y_test.values
                }
                
                st.session_state.evaluation_results = evaluation_results
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("MAE", f"{mae:.4f}")
                
                with col2:
                    st.metric("RMSE", f"{rmse:.4f}")
                
                with col3:
                    st.metric("R² Score", f"{r2:.4f}")
                
                with col4:
                    st.metric("Accuracy", f"{accuracy:.4f}")
                
                st.success("Model evaluation completed!")
                
            except Exception as e:
                st.error(f"Error evaluating model: {str(e)}")
    
    # Visualizations
    if st.session_state.evaluation_results:
        st.markdown("### Evaluation Visualizations")
        
        results = st.session_state.evaluation_results
        y_true = results['true_values']
        y_pred = results['predictions']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction vs Actual
            fig = px.scatter(x=y_true, y=y_pred, 
                           title='Predicted vs Actual Satisfaction Scores',
                           labels={'x': 'Actual', 'y': 'Predicted'})
            
            # Add perfect prediction line
            fig.add_trace(go.Scatter(x=[1, 5], y=[1, 5], mode='lines', 
                                   name='Perfect Prediction', line=dict(dash='dash')))
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Residuals plot
            residuals = y_true - y_pred
            fig = px.scatter(x=y_pred, y=residuals, 
                           title='Residuals Plot',
                           labels={'x': 'Predicted', 'y': 'Residuals'})
            
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution comparison
        st.markdown("#### Distribution Comparison")
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=y_true, name='Actual', opacity=0.7))
        fig.add_trace(go.Histogram(x=y_pred, name='Predicted', opacity=0.7))
        fig.update_layout(title='Distribution Comparison', barmode='overlay')
        
        st.plotly_chart(fig, use_container_width=True)

def show_insights_page():
    """Display the insights page."""
    st.markdown("## 🔍 Business Insights")
    
    if not st.session_state.data_generated:
        st.warning("Please generate data first from the Home page.")
        return
    
    data = st.session_state.data
    
    # Business metrics
    st.markdown("### Key Business Metrics")
    
    # Calculate business metrics
    from src.eval import BusinessMetricsCalculator
    
    business_calc = BusinessMetricsCalculator()
    
    # Satisfaction trends
    trends = business_calc.calculate_satisfaction_trends(data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Satisfaction", f"{trends['overall_mean']:.2f}")
    
    with col2:
        st.metric("Trend Slope", f"{trends['trend_slope']:.4f}")
    
    with col3:
        st.metric("Total Responses", trends['total_responses'])
    
    # Department comparison
    dept_comparison = business_calc.calculate_department_comparison(data)
    
    st.markdown("#### Department Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"🏆 Best Department: {dept_comparison['best_department']}")
    
    with col2:
        st.warning(f"⚠️ Needs Attention: {dept_comparison['worst_department']}")
    
    # At-risk analysis
    at_risk = business_calc.identify_at_risk_employees(data)
    
    st.markdown("#### At-Risk Employees")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("At-Risk Count", at_risk['current_at_risk'])
    
    with col2:
        st.metric("At-Risk Rate", f"{at_risk['current_at_risk_rate']:.1%}")
    
    # Actionable insights
    insights = business_calc.calculate_actionable_insights(data)
    
    st.markdown("### 💡 Actionable Insights")
    
    for i, insight in enumerate(insights[:5]):  # Show top 5 insights
        priority_color = "🔴" if insight['priority'] == 'high' else "🟡"
        
        with st.expander(f"{priority_color} {insight['title']}"):
            st.write(f"**Description:** {insight['description']}")
            st.write(f"**Action:** {insight['action']}")
            st.write(f"**Impact:** {insight['impact']}")
    
    # Risk factors
    st.markdown("#### Risk Factors by Group")
    
    for factor, risk_by_group in at_risk['risk_factors'].items():
        st.markdown(f"**{factor.title()}:**")
        
        risk_df = pd.DataFrame({
            'Group': risk_by_group.index,
            'Risk Rate': risk_by_group.values
        }).sort_values('Risk Rate', ascending=False)
        
        fig = px.bar(risk_df, x='Risk Rate', y='Group', 
                     orientation='h', title=f'Risk Rate by {factor.title()}')
        st.plotly_chart(fig, use_container_width=True)

def show_settings_page():
    """Display the settings page."""
    st.markdown("## ⚙️ Settings")
    
    st.markdown("### Configuration")
    
    # Data generation settings
    st.markdown("#### Data Generation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_samples = st.number_input("Number of Samples", min_value=100, max_value=5000, value=1000)
    
    with col2:
        satisfaction_dist = st.selectbox(
            "Satisfaction Distribution",
            ["Realistic", "Optimistic", "Pessimistic"],
            help="Choose the distribution of satisfaction levels"
        )
    
    # Model settings
    st.markdown("#### Model Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        random_state = st.number_input("Random State", min_value=0, max_value=1000, value=42)
    
    with col2:
        enable_fairness = st.checkbox("Enable Fairness Analysis", value=True)
    
    # Display current settings
    st.markdown("### Current Settings")
    
    settings = {
        "Data Samples": n_samples,
        "Satisfaction Distribution": satisfaction_dist,
        "Random State": random_state,
        "Fairness Analysis": enable_fairness
    }
    
    for key, value in settings.items():
        st.write(f"**{key}:** {value}")
    
    # Reset button
    if st.button("🔄 Reset All Data", type="secondary"):
        st.session_state.data_generated = False
        st.session_state.data = None
        st.session_state.model_trained = False
        st.session_state.model = None
        st.session_state.evaluation_results = None
        st.success("All data has been reset!")
        st.rerun()

if __name__ == "__main__":
    main()
