#!/usr/bin/env python3
"""Evaluation script for employee satisfaction analysis.

This script evaluates trained models and generates comprehensive reports.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import yaml
import joblib
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import SatisfactionPredictor
from eval import ModelEvaluator, VisualizationGenerator, ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)


def load_data(data_path: str) -> pd.DataFrame:
    """Load employee data."""
    try:
        return pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)


def load_model(model_path: str) -> SatisfactionPredictor:
    """Load trained model."""
    try:
        model = SatisfactionPredictor()
        model.load_model(model_path)
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate employee satisfaction models")
    parser.add_argument("--model-path", default="assets/models/satisfaction_model.joblib",
                       help="Path to trained model")
    parser.add_argument("--data-path", default="data/employee_data.csv",
                       help="Path to employee data")
    parser.add_argument("--eval-config", default="configs/eval_config.yaml",
                       help="Path to evaluation config")
    parser.add_argument("--output-dir", default="assets/reports",
                       help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Load configuration
    eval_config = load_config(args.eval_config)
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    data = load_data(args.data_path)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = load_model(args.model_path)
    
    # Prepare test data (using last 20% as test set)
    from sklearn.model_selection import train_test_split
    from features import FeatureEngineer
    
    # Feature engineering
    feature_engineer = FeatureEngineer()
    data_processed = feature_engineer.fit_transform(data)
    
    X = data_processed.drop(['satisfaction_score', 'satisfaction_level', 'employee_id'], axis=1, errors='ignore')
    y = data_processed['satisfaction_score']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Prepare sensitive features for fairness analysis
    sensitive_features = data[['age_group', 'gender', 'department', 'role_level']].iloc[X_test.index]
    
    # Initialize evaluator
    evaluator = ModelEvaluator(eval_config)
    
    # Perform comprehensive evaluation
    logger.info("Performing comprehensive evaluation...")
    evaluation_results = evaluator.evaluate_comprehensive(
        model, X_test, y_test, sensitive_features
    )
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    viz_generator = VisualizationGenerator()
    
    # Plot satisfaction distribution
    viz_generator.plot_satisfaction_distribution(
        data, 
        save_path=os.path.join(args.output_dir, "satisfaction_distribution.png")
    )
    
    # Plot model performance
    viz_generator.plot_model_performance(
        evaluation_results,
        save_path=os.path.join(args.output_dir, "model_performance.png")
    )
    
    # Plot fairness analysis
    if evaluation_results['fairness_results']:
        viz_generator.plot_fairness_analysis(
            evaluation_results['fairness_results'],
            save_path=os.path.join(args.output_dir, "fairness_analysis.png")
        )
    
    # Generate HTML report
    logger.info("Generating HTML report...")
    report_generator = ReportGenerator(args.output_dir)
    report_path = report_generator.generate_html_report(evaluation_results)
    
    # Print evaluation summary
    logger.info("Evaluation completed successfully!")
    logger.info("=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    
    # ML Metrics
    ml_metrics = evaluation_results['ml_metrics']
    logger.info("ML Metrics:")
    logger.info(f"  MAE: {ml_metrics['regression']['mae']:.4f}")
    logger.info(f"  RMSE: {ml_metrics['regression']['rmse']:.4f}")
    logger.info(f"  R²: {ml_metrics['regression']['r2']:.4f}")
    logger.info(f"  Accuracy: {ml_metrics['classification']['accuracy']:.4f}")
    
    # Business Metrics
    business_metrics = evaluation_results['business_metrics']
    logger.info("\nBusiness Metrics:")
    logger.info(f"  Overall Satisfaction: {business_metrics['satisfaction_trends']['overall_mean']:.2f}")
    logger.info(f"  At-Risk Rate: {business_metrics['at_risk_analysis']['current_at_risk_rate']:.1%}")
    logger.info(f"  Best Department: {business_metrics['department_comparison']['best_department']}")
    logger.info(f"  Worst Department: {business_metrics['department_comparison']['worst_department']}")
    
    # Fairness Results
    if evaluation_results['fairness_results']:
        logger.info("\nFairness Analysis:")
        for attr, results in evaluation_results['fairness_results'].items():
            dp_diff = results.get('demographic_parity_difference', 'N/A')
            logger.info(f"  {attr.title()} - Demographic Parity Difference: {dp_diff:.4f}" if isinstance(dp_diff, (int, float)) else f"  {attr.title()} - Demographic Parity Difference: {dp_diff}")
    
    logger.info(f"\nReport generated: {report_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
