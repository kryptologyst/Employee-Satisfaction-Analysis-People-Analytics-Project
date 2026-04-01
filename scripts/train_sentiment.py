#!/usr/bin/env python3
"""Training script for employee satisfaction analysis.

This script trains sentiment analysis and satisfaction prediction models
using the configuration files.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import yaml
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import EmployeeDataGenerator, DataProcessor
from features import FeatureEngineer
from models import SentimentAnalyzer, SatisfactionPredictor, ModelEvaluator
from eval import BusinessMetricsCalculator, ReportGenerator

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


def train_sentiment_model(config: dict, data: pd.DataFrame) -> SentimentAnalyzer:
    """Train sentiment analysis model."""
    logger.info("Training sentiment analysis model...")
    
    sentiment_analyzer = SentimentAnalyzer(config)
    
    # Test on sample data
    sample_texts = data['feedback_text'].head(10).tolist()
    results = sentiment_analyzer.analyze_batch(sample_texts)
    
    logger.info(f"Sentiment analysis completed on {len(sample_texts)} samples")
    return sentiment_analyzer


def train_satisfaction_model(config: dict, data: pd.DataFrame) -> SatisfactionPredictor:
    """Train satisfaction prediction model."""
    logger.info("Training satisfaction prediction model...")
    
    # Feature engineering
    feature_engineer = FeatureEngineer(config.get('features', {}))
    data_processed = feature_engineer.fit_transform(data)
    
    # Prepare features and target
    X = data_processed.drop(['satisfaction_score', 'satisfaction_level', 'employee_id'], axis=1, errors='ignore')
    y = data_processed['satisfaction_score']
    
    # Train-test split
    test_size = config.get('data_split', {}).get('test_size', 0.2)
    random_state = config.get('data_split', {}).get('random_state', 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train model
    model = SatisfactionPredictor(config)
    model.fit(X_train, y_train)
    
    logger.info(f"Satisfaction model trained on {len(X_train)} samples")
    
    # Save model
    model_path = "assets/models/satisfaction_model.joblib"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    
    return model, X_test, y_test


def evaluate_model(model: SatisfactionPredictor, X_test: pd.DataFrame, y_test: pd.Series, config: dict):
    """Evaluate the trained model."""
    logger.info("Evaluating model performance...")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config.get('evaluation', {}))
    
    # Perform evaluation
    evaluation_results = evaluator.evaluate_comprehensive(
        model, X_test, y_test
    )
    
    # Generate report
    report_generator = ReportGenerator()
    report_path = report_generator.generate_html_report(evaluation_results)
    
    logger.info(f"Evaluation report generated: {report_path}")
    
    return evaluation_results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train employee satisfaction models")
    parser.add_argument("--sentiment-config", default="configs/sentiment_config.yaml",
                       help="Path to sentiment analysis config")
    parser.add_argument("--satisfaction-config", default="configs/satisfaction_config.yaml",
                       help="Path to satisfaction prediction config")
    parser.add_argument("--eval-config", default="configs/eval_config.yaml",
                       help="Path to evaluation config")
    parser.add_argument("--n-samples", type=int, default=1000,
                       help="Number of samples to generate")
    parser.add_argument("--output-dir", default="assets/models",
                       help="Output directory for models")
    
    args = parser.parse_args()
    
    # Load configurations
    sentiment_config = load_config(args.sentiment_config)
    satisfaction_config = load_config(args.satisfaction_config)
    eval_config = load_config(args.eval_config)
    
    # Generate synthetic data
    logger.info(f"Generating {args.n_samples} employee records...")
    generator = EmployeeDataGenerator()
    data = generator.generate_employee_data(n_samples=args.n_samples)
    
    # Save data
    data_path = "data/employee_data.csv"
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    data.to_csv(data_path, index=False)
    logger.info(f"Data saved to {data_path}")
    
    # Train sentiment model
    sentiment_model = train_sentiment_model(sentiment_config, data)
    
    # Save sentiment model
    sentiment_path = os.path.join(args.output_dir, "sentiment_model.joblib")
    os.makedirs(args.output_dir, exist_ok=True)
    joblib.dump(sentiment_model, sentiment_path)
    logger.info(f"Sentiment model saved to {sentiment_path}")
    
    # Train satisfaction model
    satisfaction_model, X_test, y_test = train_satisfaction_model(satisfaction_config, data)
    
    # Evaluate model
    evaluation_results = evaluate_model(satisfaction_model, X_test, y_test, eval_config)
    
    # Print summary
    logger.info("Training completed successfully!")
    logger.info(f"Model performance - MAE: {evaluation_results['ml_metrics']['regression']['mae']:.4f}")
    logger.info(f"Model performance - R²: {evaluation_results['ml_metrics']['regression']['r2']:.4f}")


if __name__ == "__main__":
    main()
