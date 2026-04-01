"""Tests for employee satisfaction analysis.

This module contains unit tests for the employee satisfaction analysis system.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import EmployeeDataGenerator, DataProcessor
from features import TextFeatureExtractor, FeatureEngineer
from models import SentimentAnalyzer, SatisfactionPredictor
from eval import BusinessMetricsCalculator, ModelEvaluator


class TestEmployeeDataGenerator:
    """Test cases for EmployeeDataGenerator."""
    
    def test_init(self):
        """Test generator initialization."""
        generator = EmployeeDataGenerator(random_state=42)
        assert generator.random_state == 42
        assert len(generator.departments) > 0
        assert len(generator.role_levels) > 0
    
    def test_generate_employee_data(self):
        """Test employee data generation."""
        generator = EmployeeDataGenerator(random_state=42)
        data = generator.generate_employee_data(n_samples=100)
        
        assert len(data) == 100
        assert 'employee_id' in data.columns
        assert 'satisfaction_score' in data.columns
        assert 'feedback_text' in data.columns
        assert data['satisfaction_score'].min() >= 1.0
        assert data['satisfaction_score'].max() <= 5.0
    
    def test_generate_survey_responses(self):
        """Test survey response generation."""
        generator = EmployeeDataGenerator(random_state=42)
        employee_data = generator.generate_employee_data(n_samples=10)
        responses = generator.generate_survey_responses(employee_data, n_responses_per_employee=2)
        
        assert len(responses) == 20  # 10 employees * 2 responses each
        assert 'response_id' in responses.columns
        assert 'employee_id' in responses.columns
        assert 'rating_response' in responses.columns


class TestDataProcessor:
    """Test cases for DataProcessor."""
    
    def test_init(self):
        """Test processor initialization."""
        processor = DataProcessor()
        assert processor.logger is not None
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        processor = DataProcessor()
        
        # Test normal text
        result = processor.preprocess_text("Hello World!")
        assert result == "hello world!"
        
        # Test empty text
        result = processor.preprocess_text("")
        assert result == ""
        
        # Test None input
        result = processor.preprocess_text(None)
        assert result == ""
    
    def test_create_train_test_split(self):
        """Test train-test split creation."""
        processor = DataProcessor()
        
        # Create sample data
        data = pd.DataFrame({
            'id': range(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        train, val, test = processor.create_train_test_split(
            data, test_size=0.2, validation_size=0.2, stratify_column='target'
        )
        
        assert len(train) + len(val) + len(test) == 100
        assert len(test) == 20  # 20% of 100
        assert len(val) == 20   # 20% of remaining 80


class TestTextFeatureExtractor:
    """Test cases for TextFeatureExtractor."""
    
    def test_init(self):
        """Test extractor initialization."""
        extractor = TextFeatureExtractor()
        assert extractor.config == {}
        assert extractor.vader_analyzer is not None
    
    def test_extract_sentiment_features(self):
        """Test sentiment feature extraction."""
        extractor = TextFeatureExtractor()
        texts = ["I love this!", "This is terrible.", "It's okay."]
        
        features = extractor.extract_sentiment_features(texts)
        
        assert len(features) == 3
        assert 'textblob_polarity' in features.columns
        assert 'vader_compound' in features.columns
    
    def test_extract_linguistic_features(self):
        """Test linguistic feature extraction."""
        extractor = TextFeatureExtractor()
        texts = ["Hello world!", "This is a test.", ""]
        
        features = extractor.extract_linguistic_features(texts)
        
        assert len(features) == 3
        assert 'text_length' in features.columns
        assert 'word_count' in features.columns
        assert features.iloc[2]['text_length'] == 0  # Empty text


class TestSentimentAnalyzer:
    """Test cases for SentimentAnalyzer."""
    
    def test_init(self):
        """Test analyzer initialization."""
        config = {
            'traditional': {'textblob': {'enabled': True}},
            'ensemble': {'weights': {'textblob': 1.0}}
        }
        analyzer = SentimentAnalyzer(config)
        
        assert 'textblob' in analyzer.models
        assert analyzer.ensemble_weights['textblob'] == 1.0
    
    def test_analyze_sentiment(self):
        """Test sentiment analysis."""
        config = {
            'traditional': {'textblob': {'enabled': True}},
            'ensemble': {'weights': {'textblob': 1.0}}
        }
        analyzer = SentimentAnalyzer(config)
        
        result = analyzer.analyze_sentiment("I love this!")
        
        assert 'sentiment' in result
        assert 'confidence' in result
        assert 'polarity' in result
        assert result['sentiment'] in ['positive', 'negative', 'neutral']
    
    def test_analyze_batch(self):
        """Test batch sentiment analysis."""
        config = {
            'traditional': {'textblob': {'enabled': True}},
            'ensemble': {'weights': {'textblob': 1.0}}
        }
        analyzer = SentimentAnalyzer(config)
        
        texts = ["I love this!", "This is terrible.", "It's okay."]
        results = analyzer.analyze_batch(texts)
        
        assert len(results) == 3
        assert all('sentiment' in result for result in results)


class TestSatisfactionPredictor:
    """Test cases for SatisfactionPredictor."""
    
    def test_init(self):
        """Test predictor initialization."""
        config = {'model': {'type': 'gradient_boosting'}}
        predictor = SatisfactionPredictor(config)
        
        assert predictor.model is not None
        assert not predictor.is_fitted
    
    def test_fit_and_predict(self):
        """Test model fitting and prediction."""
        config = {'model': {'type': 'gradient_boosting'}}
        predictor = SatisfactionPredictor(config)
        
        # Create sample data
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y = pd.Series(np.random.uniform(1, 5, 100))
        
        # Fit model
        predictor.fit(X, y)
        assert predictor.is_fitted
        
        # Make predictions
        predictions = predictor.predict(X)
        assert len(predictions) == 100
        assert all(1.0 <= pred <= 5.0 for pred in predictions)
    
    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        config = {'model': {'type': 'gradient_boosting'}}
        predictor = SatisfactionPredictor(config)
        
        # Create sample data
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y = pd.Series(np.random.uniform(1, 5, 100))
        
        # Fit model
        predictor.fit(X, y)
        
        # Get feature importance
        importance = predictor.get_feature_importance()
        assert len(importance) == 2
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns


class TestBusinessMetricsCalculator:
    """Test cases for BusinessMetricsCalculator."""
    
    def test_init(self):
        """Test calculator initialization."""
        calculator = BusinessMetricsCalculator()
        assert calculator.thresholds['high_satisfaction'] == 4.0
    
    def test_calculate_department_comparison(self):
        """Test department comparison calculation."""
        calculator = BusinessMetricsCalculator()
        
        # Create sample data
        data = pd.DataFrame({
            'department': ['A', 'A', 'B', 'B'],
            'satisfaction_score': [4.0, 3.0, 5.0, 2.0],
            'satisfaction_level': ['positive', 'neutral', 'positive', 'negative']
        })
        
        result = calculator.calculate_department_comparison(data)
        
        assert 'best_department' in result
        assert 'worst_department' in result
        assert result['best_department'] == 'B'  # Higher average
        assert result['worst_department'] == 'A'  # Lower average
    
    def test_identify_at_risk_employees(self):
        """Test at-risk employee identification."""
        calculator = BusinessMetricsCalculator()
        
        # Create sample data
        data = pd.DataFrame({
            'employee_id': ['E1', 'E2', 'E3', 'E4'],
            'satisfaction_score': [1.5, 2.0, 4.0, 5.0],
            'department': ['A', 'A', 'B', 'B']
        })
        
        result = calculator.identify_at_risk_employees(data)
        
        assert result['current_at_risk'] == 2  # E1 and E2 below 2.5
        assert result['current_at_risk_rate'] == 0.5  # 2 out of 4


class TestModelEvaluator:
    """Test cases for ModelEvaluator."""
    
    def test_init(self):
        """Test evaluator initialization."""
        evaluator = ModelEvaluator()
        assert evaluator.business_calculator is not None
    
    def test_calculate_ml_metrics(self):
        """Test ML metrics calculation."""
        evaluator = ModelEvaluator()
        
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        
        metrics = evaluator._calculate_ml_metrics(y_true, y_pred)
        
        assert 'regression' in metrics
        assert 'classification' in metrics
        assert 'mae' in metrics['regression']
        assert 'accuracy' in metrics['classification']


# Integration tests
class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Generate data
        generator = EmployeeDataGenerator(random_state=42)
        data = generator.generate_employee_data(n_samples=50)
        
        # Feature engineering
        feature_engineer = FeatureEngineer()
        data_processed = feature_engineer.fit_transform(data)
        
        # Prepare features
        X = data_processed.drop(['satisfaction_score', 'satisfaction_level', 'employee_id'], axis=1, errors='ignore')
        y = data_processed['satisfaction_score']
        
        # Train model
        config = {'model': {'type': 'gradient_boosting'}}
        model = SatisfactionPredictor(config)
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Verify results
        assert len(predictions) == len(X)
        assert all(1.0 <= pred <= 5.0 for pred in predictions)
        
        # Test evaluation
        evaluator = ModelEvaluator()
        evaluation_results = evaluator.evaluate_comprehensive(model, X, y)
        
        assert 'ml_metrics' in evaluation_results
        assert 'business_metrics' in evaluation_results


if __name__ == "__main__":
    pytest.main([__file__])
