"""Evaluation module for employee satisfaction analysis.

This module provides comprehensive evaluation capabilities including
business metrics, fairness analysis, and explainability features.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.calibration import calibration_curve
import shap
import joblib
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class BusinessMetricsCalculator:
    """Calculate business-relevant metrics for employee satisfaction."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize business metrics calculator.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        self.thresholds = self.config.get('thresholds', {
            'high_satisfaction': 4.0,
            'low_satisfaction': 2.5,
            'risk_threshold': 0.3
        })
    
    def calculate_satisfaction_trends(
        self, 
        df: pd.DataFrame, 
        date_column: str = 'response_date'
    ) -> Dict[str, Any]:
        """Calculate satisfaction trends over time.
        
        Args:
            df: DataFrame with satisfaction data.
            date_column: Name of the date column.
            
        Returns:
            Dictionary with trend metrics.
        """
        df_with_date = df.copy()
        df_with_date[date_column] = pd.to_datetime(df_with_date[date_column])
        df_with_date['month'] = df_with_date[date_column].dt.to_period('M')
        
        monthly_stats = df_with_date.groupby('month').agg({
            'satisfaction_score': ['mean', 'std', 'count'],
            'satisfaction_level': lambda x: (x == 'positive').mean()
        }).round(3)
        
        # Flatten column names
        monthly_stats.columns = ['mean_satisfaction', 'std_satisfaction', 'count', 'positive_rate']
        
        # Calculate trend
        if len(monthly_stats) > 1:
            trend_slope = np.polyfit(range(len(monthly_stats)), monthly_stats['mean_satisfaction'], 1)[0]
        else:
            trend_slope = 0
        
        return {
            'monthly_stats': monthly_stats,
            'trend_slope': trend_slope,
            'overall_mean': df['satisfaction_score'].mean(),
            'overall_std': df['satisfaction_score'].std(),
            'total_responses': len(df)
        }
    
    def calculate_department_comparison(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compare satisfaction across departments.
        
        Args:
            df: DataFrame with satisfaction data.
            
        Returns:
            Dictionary with department comparison metrics.
        """
        dept_stats = df.groupby('department').agg({
            'satisfaction_score': ['mean', 'std', 'count'],
            'satisfaction_level': lambda x: (x == 'positive').mean()
        }).round(3)
        
        dept_stats.columns = ['mean_satisfaction', 'std_satisfaction', 'count', 'positive_rate']
        
        # Identify best and worst performing departments
        best_dept = dept_stats['mean_satisfaction'].idxmax()
        worst_dept = dept_stats['mean_satisfaction'].idxmin()
        
        return {
            'department_stats': dept_stats,
            'best_department': best_dept,
            'worst_department': worst_dept,
            'department_range': dept_stats['mean_satisfaction'].max() - dept_stats['mean_satisfaction'].min()
        }
    
    def identify_at_risk_employees(
        self, 
        df: pd.DataFrame, 
        predictions: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Identify employees at risk of dissatisfaction.
        
        Args:
            df: DataFrame with employee data.
            predictions: Optional model predictions.
            
        Returns:
            Dictionary with risk analysis results.
        """
        if predictions is not None:
            df['predicted_satisfaction'] = predictions
        
        # Define risk criteria
        risk_threshold = self.thresholds['low_satisfaction']
        
        # Current risk (based on actual scores)
        current_risk = df[df['satisfaction_score'] < risk_threshold]
        
        # Predicted risk (if predictions available)
        predicted_risk = None
        if predictions is not None:
            predicted_risk = df[df['predicted_satisfaction'] < risk_threshold]
        
        # Risk factors analysis
        risk_factors = {}
        for col in ['department', 'role_level', 'age_group', 'gender']:
            if col in df.columns:
                risk_by_group = df.groupby(col).apply(
                    lambda x: (x['satisfaction_score'] < risk_threshold).mean()
                ).sort_values(ascending=False)
                risk_factors[col] = risk_by_group
        
        return {
            'current_at_risk': len(current_risk),
            'current_at_risk_rate': len(current_risk) / len(df),
            'predicted_at_risk': len(predicted_risk) if predicted_risk is not None else None,
            'predicted_at_risk_rate': len(predicted_risk) / len(df) if predicted_risk is not None else None,
            'risk_factors': risk_factors,
            'at_risk_employees': current_risk[['employee_id', 'department', 'role_level', 'satisfaction_score']].to_dict('records')
        }
    
    def calculate_actionable_insights(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate actionable insights from satisfaction data.
        
        Args:
            df: DataFrame with satisfaction data.
            
        Returns:
            List of actionable insights.
        """
        insights = []
        
        # Department insights
        dept_stats = df.groupby('department')['satisfaction_score'].agg(['mean', 'count'])
        low_satisfaction_depts = dept_stats[dept_stats['mean'] < self.thresholds['high_satisfaction']]
        
        for dept in low_satisfaction_depts.index:
            insights.append({
                'type': 'department_focus',
                'priority': 'high',
                'title': f'Focus on {dept} Department',
                'description': f'{dept} has below-average satisfaction ({low_satisfaction_depts.loc[dept, "mean"]:.2f})',
                'action': f'Conduct targeted survey and focus groups in {dept}',
                'impact': f'Potential to improve satisfaction for {low_satisfaction_depts.loc[dept, "count"]} employees'
            })
        
        # Role level insights
        role_stats = df.groupby('role_level')['satisfaction_score'].agg(['mean', 'count'])
        low_satisfaction_roles = role_stats[role_stats['mean'] < self.thresholds['high_satisfaction']]
        
        for role in low_satisfaction_roles.index:
            insights.append({
                'type': 'role_level_focus',
                'priority': 'medium',
                'title': f'Address {role} Level Concerns',
                'description': f'{role} level employees show lower satisfaction ({low_satisfaction_roles.loc[role, "mean"]:.2f})',
                'action': f'Review career development and recognition programs for {role} level',
                'impact': f'Potential to improve satisfaction for {low_satisfaction_roles.loc[role, "count"]} employees'
            })
        
        # Tenure insights
        df['tenure_years'] = df['tenure_months'] / 12
        tenure_bins = pd.cut(df['tenure_years'], bins=[0, 1, 3, 5, 10, 100], labels=['<1yr', '1-3yr', '3-5yr', '5-10yr', '10+yr'])
        tenure_stats = df.groupby(tenure_bins)['satisfaction_score'].agg(['mean', 'count'])
        
        low_satisfaction_tenure = tenure_stats[tenure_stats['mean'] < self.thresholds['high_satisfaction']]
        for tenure in low_satisfaction_tenure.index:
            insights.append({
                'type': 'tenure_focus',
                'priority': 'medium',
                'title': f'Support {tenure} Tenure Employees',
                'description': f'Employees with {tenure} tenure show lower satisfaction ({low_satisfaction_tenure.loc[tenure, "mean"]:.2f})',
                'action': f'Implement onboarding improvements and retention programs for {tenure} employees',
                'impact': f'Potential to improve satisfaction for {low_satisfaction_tenure.loc[tenure, "count"]} employees'
            })
        
        return insights


class ModelEvaluator:
    """Comprehensive model evaluation with business context."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize model evaluator.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        self.business_calculator = BusinessMetricsCalculator(config.get('business_metrics', {}))
    
    def evaluate_comprehensive(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sensitive_features: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive model evaluation.
        
        Args:
            model: Trained model.
            X_test: Test features.
            y_test: Test targets.
            sensitive_features: Protected attributes.
            
        Returns:
            Dictionary with comprehensive evaluation results.
        """
        logger.info("Starting comprehensive model evaluation...")
        
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # ML Metrics
        ml_metrics = self._calculate_ml_metrics(y_test, y_pred)
        
        # Business Metrics
        test_df = X_test.copy()
        test_df['satisfaction_score'] = y_test
        test_df['predicted_satisfaction'] = y_pred
        
        business_metrics = {
            'satisfaction_trends': self.business_calculator.calculate_satisfaction_trends(test_df),
            'department_comparison': self.business_calculator.calculate_department_comparison(test_df),
            'at_risk_analysis': self.business_calculator.identify_at_risk_employees(test_df, y_pred),
            'actionable_insights': self.business_calculator.calculate_actionable_insights(test_df)
        }
        
        # Fairness Analysis
        fairness_results = {}
        if sensitive_features is not None:
            fairness_results = self._analyze_fairness(y_test, y_pred, sensitive_features)
        
        # Feature Importance
        feature_importance = model.get_feature_importance() if hasattr(model, 'get_feature_importance') else None
        
        # Calibration Analysis
        calibration_results = self._analyze_calibration(y_test, y_pred)
        
        return {
            'ml_metrics': ml_metrics,
            'business_metrics': business_metrics,
            'fairness_results': fairness_results,
            'feature_importance': feature_importance,
            'calibration_results': calibration_results,
            'predictions': y_pred,
            'true_values': y_test.values,
            'test_data': test_df
        }
    
    def _calculate_ml_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate standard ML metrics.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            
        Returns:
            Dictionary with ML metrics.
        """
        # Regression metrics
        regression_metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'r2': r2_score(y_true, y_pred),
            'explained_variance': 1 - np.var(y_true - y_pred) / np.var(y_true)
        }
        
        # Classification metrics (using 3.0 as threshold)
        y_true_binary = (y_true >= 3.0).astype(int)
        y_pred_binary = (y_pred >= 3.0).astype(int)
        
        classification_metrics = {
            'accuracy': accuracy_score(y_true_binary, y_pred_binary),
            'precision': precision_score(y_true_binary, y_pred_binary, average='weighted'),
            'recall': recall_score(y_true_binary, y_pred_binary, average='weighted'),
            'f1_score': f1_score(y_true_binary, y_pred_binary, average='weighted')
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        
        return {
            'regression': regression_metrics,
            'classification': classification_metrics,
            'confusion_matrix': cm
        }
    
    def _analyze_fairness(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        sensitive_features: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze fairness across protected attributes.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            sensitive_features: Protected attributes.
            
        Returns:
            Dictionary with fairness analysis.
        """
        fairness_results = {}
        
        for attr in sensitive_features.columns:
            attr_results = {}
            
            # Group-wise performance
            groups = sensitive_features[attr]
            group_stats = {}
            
            for group in groups.unique():
                mask = groups == group
                group_true = y_true[mask]
                group_pred = y_pred[mask]
                
                group_stats[group] = {
                    'count': mask.sum(),
                    'mean_true': group_true.mean(),
                    'mean_pred': group_pred.mean(),
                    'mae': mean_absolute_error(group_true, group_pred),
                    'rmse': np.sqrt(mean_squared_error(group_true, group_pred))
                }
            
            attr_results['group_stats'] = group_stats
            
            # Calculate fairness metrics
            y_true_binary = (y_true >= 3.0).astype(int)
            y_pred_binary = (y_pred >= 3.0).astype(int)
            
            # Demographic parity
            positive_rate_by_group = {}
            for group in groups.unique():
                mask = groups == group
                positive_rate_by_group[group] = y_pred_binary[mask].mean()
            
            dp_diff = max(positive_rate_by_group.values()) - min(positive_rate_by_group.values())
            attr_results['demographic_parity_difference'] = dp_diff
            
            fairness_results[attr] = attr_results
        
        return fairness_results
    
    def _analyze_calibration(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction calibration.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            
        Returns:
            Dictionary with calibration analysis.
        """
        # Convert to binary for calibration analysis
        y_true_binary = (y_true >= 3.0).astype(int)
        
        # Create probability-like scores
        y_prob = np.clip((y_pred - 1) / 4, 0, 1)  # Scale to [0, 1]
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_binary, y_prob, n_bins=10
        )
        
        # Calculate calibration error
        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        
        return {
            'calibration_error': calibration_error,
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value
        }


class VisualizationGenerator:
    """Generate visualizations for evaluation results."""
    
    def __init__(self, output_dir: str = "assets/plots") -> None:
        """Initialize visualization generator.
        
        Args:
            output_dir: Directory to save plots.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_satisfaction_distribution(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """Plot satisfaction score distribution.
        
        Args:
            df: DataFrame with satisfaction data.
            save_path: Optional path to save the plot.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Overall distribution
        axes[0, 0].hist(df['satisfaction_score'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Overall Satisfaction Score Distribution')
        axes[0, 0].set_xlabel('Satisfaction Score')
        axes[0, 0].set_ylabel('Frequency')
        
        # By department
        dept_satisfaction = df.groupby('department')['satisfaction_score'].mean().sort_values(ascending=True)
        axes[0, 1].barh(dept_satisfaction.index, dept_satisfaction.values)
        axes[0, 1].set_title('Average Satisfaction by Department')
        axes[0, 1].set_xlabel('Average Satisfaction Score')
        
        # By role level
        role_satisfaction = df.groupby('role_level')['satisfaction_score'].mean().sort_values(ascending=True)
        axes[1, 0].barh(role_satisfaction.index, role_satisfaction.values)
        axes[1, 0].set_title('Average Satisfaction by Role Level')
        axes[1, 0].set_xlabel('Average Satisfaction Score')
        
        # Satisfaction level pie chart
        satisfaction_counts = df['satisfaction_level'].value_counts()
        axes[1, 1].pie(satisfaction_counts.values, labels=satisfaction_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Satisfaction Level Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_performance(self, evaluation_results: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """Plot model performance metrics.
        
        Args:
            evaluation_results: Results from model evaluation.
            save_path: Optional path to save the plot.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Prediction vs Actual scatter plot
        y_true = evaluation_results['true_values']
        y_pred = evaluation_results['predictions']
        
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([1, 5], [1, 5], 'r--', label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual Satisfaction Score')
        axes[0, 0].set_ylabel('Predicted Satisfaction Score')
        axes[0, 0].set_title('Prediction vs Actual')
        axes[0, 0].legend()
        
        # Residuals plot
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Satisfaction Score')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        
        # Feature importance (if available)
        if evaluation_results.get('feature_importance') is not None:
            top_features = evaluation_results['feature_importance'].head(10)
            axes[1, 0].barh(top_features['feature'], top_features['importance'])
            axes[1, 0].set_title('Top 10 Feature Importance')
            axes[1, 0].set_xlabel('Importance')
        
        # Confusion matrix
        cm = evaluation_results['ml_metrics']['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title('Confusion Matrix (Threshold: 3.0)')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_fairness_analysis(self, fairness_results: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """Plot fairness analysis results.
        
        Args:
            fairness_results: Results from fairness analysis.
            save_path: Optional path to save the plot.
        """
        n_attrs = len(fairness_results)
        if n_attrs == 0:
            return
        
        fig, axes = plt.subplots(2, n_attrs, figsize=(5*n_attrs, 10))
        if n_attrs == 1:
            axes = axes.reshape(2, 1)
        
        for i, (attr, results) in enumerate(fairness_results.items()):
            group_stats = results['group_stats']
            
            groups = list(group_stats.keys())
            mean_true = [group_stats[g]['mean_true'] for g in groups]
            mean_pred = [group_stats[g]['mean_pred'] for g in groups]
            
            # Actual vs Predicted by group
            x = np.arange(len(groups))
            width = 0.35
            
            axes[0, i].bar(x - width/2, mean_true, width, label='Actual', alpha=0.7)
            axes[0, i].bar(x + width/2, mean_pred, width, label='Predicted', alpha=0.7)
            axes[0, i].set_xlabel(attr.title())
            axes[0, i].set_ylabel('Average Satisfaction Score')
            axes[0, i].set_title(f'Satisfaction by {attr.title()}')
            axes[0, i].set_xticks(x)
            axes[0, i].set_xticklabels(groups, rotation=45)
            axes[0, i].legend()
            
            # MAE by group
            mae_values = [group_stats[g]['mae'] for g in groups]
            axes[1, i].bar(groups, mae_values, alpha=0.7)
            axes[1, i].set_xlabel(attr.title())
            axes[1, i].set_ylabel('Mean Absolute Error')
            axes[1, i].set_title(f'Prediction Error by {attr.title()}')
            axes[1, i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class ReportGenerator:
    """Generate comprehensive evaluation reports."""
    
    def __init__(self, output_dir: str = "assets/reports") -> None:
        """Initialize report generator.
        
        Args:
            output_dir: Directory to save reports.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_html_report(self, evaluation_results: Dict[str, Any], filename: str = None) -> str:
        """Generate HTML evaluation report.
        
        Args:
            evaluation_results: Results from comprehensive evaluation.
            filename: Optional filename for the report.
            
        Returns:
            Path to the generated HTML report.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_report_{timestamp}.html"
        
        filepath = os.path.join(self.output_dir, filename)
        
        html_content = self._create_html_content(evaluation_results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {filepath}")
        return filepath
    
    def _create_html_content(self, evaluation_results: Dict[str, Any]) -> str:
        """Create HTML content for the report.
        
        Args:
            evaluation_results: Evaluation results.
            
        Returns:
            HTML content string.
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Employee Satisfaction Analysis - Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 5px; }}
                .warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; }}
                .success {{ background-color: #d4edda; border: 1px solid #c3e6cb; padding: 10px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Employee Satisfaction Analysis - Evaluation Report</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <div class="warning">
                    <strong>DISCLAIMER:</strong> This is an experimental research tool. 
                    All insights should be validated by qualified HR professionals and 
                    used only as supplementary information for human decision-making.
                </div>
            </div>
            
            <div class="section">
                <h2>Model Performance Metrics</h2>
                <div class="metric">
                    <strong>MAE:</strong> {evaluation_results['ml_metrics']['regression']['mae']:.4f}
                </div>
                <div class="metric">
                    <strong>RMSE:</strong> {evaluation_results['ml_metrics']['regression']['rmse']:.4f}
                </div>
                <div class="metric">
                    <strong>R²:</strong> {evaluation_results['ml_metrics']['regression']['r2']:.4f}
                </div>
                <div class="metric">
                    <strong>Accuracy:</strong> {evaluation_results['ml_metrics']['classification']['accuracy']:.4f}
                </div>
            </div>
            
            <div class="section">
                <h2>Business Insights</h2>
                <h3>Satisfaction Trends</h3>
                <p>Overall satisfaction: {evaluation_results['business_metrics']['satisfaction_trends']['overall_mean']:.2f}</p>
                <p>Trend slope: {evaluation_results['business_metrics']['satisfaction_trends']['trend_slope']:.4f}</p>
                
                <h3>Department Comparison</h3>
                <p>Best performing department: {evaluation_results['business_metrics']['department_comparison']['best_department']}</p>
                <p>Worst performing department: {evaluation_results['business_metrics']['department_comparison']['worst_department']}</p>
                
                <h3>At-Risk Employees</h3>
                <p>Current at-risk rate: {evaluation_results['business_metrics']['at_risk_analysis']['current_at_risk_rate']:.2%}</p>
            </div>
            
            <div class="section">
                <h2>Actionable Insights</h2>
                {self._format_insights(evaluation_results['business_metrics']['actionable_insights'])}
            </div>
            
            <div class="section">
                <h2>Fairness Analysis</h2>
                {self._format_fairness_results(evaluation_results['fairness_results'])}
            </div>
            
        </body>
        </html>
        """
        
        return html
    
    def _format_insights(self, insights: List[Dict[str, Any]]) -> str:
        """Format actionable insights for HTML.
        
        Args:
            insights: List of insights.
            
        Returns:
            Formatted HTML string.
        """
        if not insights:
            return "<p>No specific insights generated.</p>"
        
        html = "<ul>"
        for insight in insights[:5]:  # Show top 5 insights
            priority_class = "warning" if insight['priority'] == 'high' else "success"
            html += f"""
            <li class="{priority_class}">
                <strong>{insight['title']}</strong><br>
                {insight['description']}<br>
                <em>Action:</em> {insight['action']}<br>
                <em>Impact:</em> {insight['impact']}
            </li>
            """
        html += "</ul>"
        
        return html
    
    def _format_fairness_results(self, fairness_results: Dict[str, Any]) -> str:
        """Format fairness results for HTML.
        
        Args:
            fairness_results: Fairness analysis results.
            
        Returns:
            Formatted HTML string.
        """
        if not fairness_results:
            return "<p>No fairness analysis performed.</p>"
        
        html = "<table>"
        html += "<tr><th>Attribute</th><th>Demographic Parity Difference</th><th>Status</th></tr>"
        
        for attr, results in fairness_results.items():
            dp_diff = results.get('demographic_parity_difference', 'N/A')
            if isinstance(dp_diff, (int, float)):
                status = "Good" if dp_diff < 0.1 else "Needs Attention"
                status_class = "success" if dp_diff < 0.1 else "warning"
            else:
                status = "N/A"
                status_class = ""
            
            html += f"""
            <tr>
                <td>{attr.title()}</td>
                <td>{dp_diff:.4f if isinstance(dp_diff, (int, float)) else dp_diff}</td>
                <td class="{status_class}">{status}</td>
            </tr>
            """
        
        html += "</table>"
        
        return html
