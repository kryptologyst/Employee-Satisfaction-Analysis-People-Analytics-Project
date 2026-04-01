#!/usr/bin/env python3
"""Complete Employee Satisfaction Analysis Demo

This script demonstrates the complete employee satisfaction analysis pipeline
from data generation to model evaluation and business insights generation.

DISCLAIMER: This is an experimental research tool. All outputs should be validated
by qualified HR professionals and used only as supplementary information for
human decision-making processes.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data import EmployeeDataGenerator
from features import FeatureEngineer
from models import SentimentAnalyzer, SatisfactionPredictor
from eval import ModelEvaluator, BusinessMetricsCalculator, ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run the complete employee satisfaction analysis pipeline."""
    
    print("=" * 60)
    print("EMPLOYEE SATISFACTION ANALYSIS - COMPLETE DEMO")
    print("=" * 60)
    print()
    print("DISCLAIMER: This is an experimental research tool.")
    print("All outputs should be validated by qualified HR professionals.")
    print("Use only as supplementary information for human decision-making.")
    print()
    
    # Step 1: Generate synthetic data
    logger.info("Step 1: Generating synthetic employee data...")
    generator = EmployeeDataGenerator(random_state=42)
    data = generator.generate_employee_data(n_samples=1000)
    
    print(f"✓ Generated {len(data)} employee records")
    print(f"  - Average satisfaction: {data['satisfaction_score'].mean():.2f}")
    print(f"  - Positive rate: {(data['satisfaction_level'] == 'positive').mean():.1%}")
    print(f"  - Departments: {data['department'].nunique()}")
    print()
    
    # Step 2: Sentiment Analysis
    logger.info("Step 2: Performing sentiment analysis...")
    sentiment_config = {
        'traditional': {'textblob': {'enabled': True}, 'vader': {'enabled': True}},
        'ensemble': {'weights': {'textblob': 0.5, 'vader': 0.5}}
    }
    
    sentiment_analyzer = SentimentAnalyzer(sentiment_config)
    sample_feedback = data['feedback_text'].head(5).tolist()
    sentiment_results = sentiment_analyzer.analyze_batch(sample_feedback)
    
    print("✓ Sentiment analysis completed")
    print("  Sample results:")
    for i, (feedback, result) in enumerate(zip(sample_feedback, sentiment_results)):
        print(f"    {i+1}. \"{feedback[:40]}...\" → {result['sentiment']} ({result['confidence']:.2f})")
    print()
    
    # Step 3: Feature Engineering
    logger.info("Step 3: Engineering features...")
    feature_engineer = FeatureEngineer()
    data_processed = feature_engineer.fit_transform(data)
    
    print(f"✓ Feature engineering completed")
    print(f"  - Original features: {data.shape[1]}")
    print(f"  - Engineered features: {data_processed.shape[1]}")
    print(f"  - New features added: {data_processed.shape[1] - data.shape[1]}")
    print()
    
    # Step 4: Model Training
    logger.info("Step 4: Training satisfaction prediction model...")
    
    # Prepare features and target
    X = data_processed.drop(['satisfaction_score', 'satisfaction_level', 'employee_id'], axis=1, errors='ignore')
    y = data_processed['satisfaction_score']
    
    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    satisfaction_config = {
        'model': {'type': 'gradient_boosting'},
        'gradient_boosting': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
    }
    
    model = SatisfactionPredictor(satisfaction_config)
    model.fit(X_train, y_train)
    
    print(f"✓ Model training completed")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")
    print(f"  - Features used: {len(model.feature_names)}")
    print(f"  - Training R²: {model.model.score(X_train, y_train):.4f}")
    print()
    
    # Step 5: Model Evaluation
    logger.info("Step 5: Evaluating model performance...")
    
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
    
    print(f"✓ Model evaluation completed")
    print(f"  - MAE: {mae:.4f}")
    print(f"  - RMSE: {rmse:.4f}")
    print(f"  - R² Score: {r2:.4f}")
    print(f"  - Accuracy: {accuracy:.4f}")
    print()
    
    # Step 6: Business Metrics
    logger.info("Step 6: Calculating business metrics...")
    
    business_calc = BusinessMetricsCalculator()
    
    # Satisfaction trends
    trends = business_calc.calculate_satisfaction_trends(data)
    
    # Department comparison
    dept_comparison = business_calc.calculate_department_comparison(data)
    
    # At-risk analysis
    at_risk = business_calc.identify_at_risk_employees(data, y_pred)
    
    # Actionable insights
    insights = business_calc.calculate_actionable_insights(data)
    
    print(f"✓ Business metrics calculated")
    print(f"  - Overall satisfaction: {trends['overall_mean']:.2f}")
    print(f"  - Best department: {dept_comparison['best_department']}")
    print(f"  - Worst department: {dept_comparison['worst_department']}")
    print(f"  - At-risk employees: {at_risk['current_at_risk']} ({at_risk['current_at_risk_rate']:.1%})")
    print(f"  - Actionable insights: {len(insights)}")
    print()
    
    # Step 7: Fairness Analysis
    logger.info("Step 7: Performing fairness analysis...")
    
    from models import FairnessAnalyzer
    
    # Prepare sensitive features
    sensitive_features = data[['age_group', 'gender', 'department', 'role_level']].iloc[X_test.index]
    
    fairness_config = {
        'protected_attributes': ['age_group', 'gender', 'department', 'role_level'],
        'metrics': {'demographic_parity': {'enabled': True}, 'equalized_odds': {'enabled': True}}
    }
    
    fairness_analyzer = FairnessAnalyzer(fairness_config)
    fairness_results = fairness_analyzer.analyze_fairness(y_test.values, y_pred, sensitive_features)
    
    print(f"✓ Fairness analysis completed")
    for attr, results in fairness_results.items():
        dp_diff = results.get('demographic_parity_difference', 'N/A')
        if isinstance(dp_diff, (int, float)):
            status = "Good" if dp_diff < 0.1 else "Needs Attention"
            print(f"  - {attr.title()}: {dp_diff:.4f} ({status})")
        else:
            print(f"  - {attr.title()}: {dp_diff}")
    print()
    
    # Step 8: Generate Report
    logger.info("Step 8: Generating comprehensive report...")
    
    # Comprehensive evaluation
    eval_config = {
        'business_metrics': {'thresholds': {'high_satisfaction': 4.0, 'low_satisfaction': 2.5}}
    }
    
    evaluator = ModelEvaluator(eval_config)
    evaluation_results = evaluator.evaluate_comprehensive(model, X_test, y_test, sensitive_features)
    
    # Generate HTML report
    report_generator = ReportGenerator()
    report_path = report_generator.generate_html_report(evaluation_results)
    
    print(f"✓ Comprehensive report generated")
    print(f"  - Report saved to: {report_path}")
    print()
    
    # Final Summary
    print("=" * 60)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("=" * 60)
    print()
    print("Key Findings:")
    print(f"• Model Performance: R² = {r2:.4f}, MAE = {mae:.4f}")
    print(f"• Overall Satisfaction: {trends['overall_mean']:.2f}/5.0")
    print(f"• Department Range: {dept_comparison['department_range']:.2f}")
    print(f"• At-Risk Employees: {at_risk['current_at_risk_rate']:.1%}")
    print(f"• Actionable Insights: {len(insights)} recommendations generated")
    print()
    print("Top Insights:")
    for i, insight in enumerate(insights[:3], 1):
        print(f"{i}. {insight['title']} ({insight['priority'].upper()} priority)")
    print()
    print("Files Generated:")
    print(f"• Data: data/employee_data.csv")
    print(f"• Model: assets/models/satisfaction_model.joblib")
    print(f"• Report: {report_path}")
    print()
    print("Next Steps:")
    print("1. Review the generated HTML report")
    print("2. Validate insights with HR professionals")
    print("3. Consider bias mitigation strategies")
    print("4. Implement human oversight for any production use")
    print()
    print("IMPORTANT REMINDER:")
    print("This is an experimental research tool. All outputs should be")
    print("validated by qualified HR professionals and used only as")
    print("supplementary information for human decision-making processes.")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
