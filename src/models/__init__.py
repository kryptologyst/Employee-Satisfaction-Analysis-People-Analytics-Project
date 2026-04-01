"""Models module for employee satisfaction analysis.

This module provides sentiment analysis and satisfaction prediction models
with fairness-aware capabilities.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import fairlearn
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
import shap
import joblib
import os

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Advanced sentiment analysis using multiple models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize sentiment analyzer.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        self.models = {}
        self.ensemble_weights = self.config.get('ensemble', {}).get('weights', {
            'textblob': 0.3,
            'vader': 0.3,
            'transformer': 0.4
        })
        self.thresholds = self.config.get('traditional', {}).get('thresholds', {
            'positive': 0.2,
            'negative': -0.2
        })
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize sentiment analysis models."""
        # Traditional models
        if self.config.get('traditional', {}).get('textblob', {}).get('enabled', True):
            self.models['textblob'] = TextBlob
        
        if self.config.get('traditional', {}).get('vader', {}).get('enabled', True):
            try:
                nltk.download('vader_lexicon', quiet=True)
                self.models['vader'] = SentimentIntensityAnalyzer()
            except:
                logger.warning("VADER sentiment analyzer not available")
        
        # Transformer model
        if self.config.get('transformer', {}).get('enabled', True):
            try:
                model_name = self.config.get('transformer', {}).get(
                    'model_name', 
                    'cardiffnlp/twitter-roberta-base-sentiment-latest'
                )
                self.models['transformer'] = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
            except Exception as e:
                logger.warning(f"Transformer model not available: {e}")
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of a single text.
        
        Args:
            text: Input text to analyze.
            
        Returns:
            Dictionary with sentiment analysis results.
        """
        if pd.isna(text) or text == "":
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'polarity': 0.0,
                'individual_scores': {}
            }
        
        individual_scores = {}
        
        # TextBlob analysis
        if 'textblob' in self.models:
            blob = self.models['textblob'](text)
            polarity = blob.sentiment.polarity
            individual_scores['textblob'] = polarity
        
        # VADER analysis
        if 'vader' in self.models:
            vader_scores = self.models['vader'].polarity_scores(text)
            individual_scores['vader'] = vader_scores['compound']
        
        # Transformer analysis
        if 'transformer' in self.models:
            try:
                result = self.models['transformer'](text)
                # Map transformer labels to polarity scores
                label_to_score = {
                    'LABEL_0': -1.0,  # Negative
                    'LABEL_1': 0.0,   # Neutral
                    'LABEL_2': 1.0    # Positive
                }
                individual_scores['transformer'] = label_to_score.get(
                    result[0]['label'], 0.0
                ) * result[0]['score']
            except Exception as e:
                logger.warning(f"Transformer analysis failed: {e}")
                individual_scores['transformer'] = 0.0
        
        # Ensemble prediction
        ensemble_score = self._ensemble_prediction(individual_scores)
        
        # Determine sentiment label
        if ensemble_score > self.thresholds['positive']:
            sentiment = 'positive'
        elif ensemble_score < self.thresholds['negative']:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'confidence': abs(ensemble_score),
            'polarity': ensemble_score,
            'individual_scores': individual_scores
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of texts to analyze.
            
        Returns:
            List of sentiment analysis results.
        """
        results = []
        for text in texts:
            results.append(self.analyze_sentiment(text))
        return results
    
    def _ensemble_prediction(self, individual_scores: Dict[str, float]) -> float:
        """Combine individual model predictions.
        
        Args:
            individual_scores: Dictionary of model scores.
            
        Returns:
            Ensemble prediction score.
        """
        weighted_sum = 0.0
        total_weight = 0.0
        
        for model_name, score in individual_scores.items():
            weight = self.ensemble_weights.get(model_name, 0.0)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0


class SatisfactionPredictor:
    """Predict employee satisfaction using machine learning models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize satisfaction predictor.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        self.model = None
        self.feature_names = []
        self.is_fitted = False
        self.scaler = None
        
        # Initialize model based on config
        model_type = self.config.get('model', {}).get('type', 'gradient_boosting')
        self._initialize_model(model_type)
    
    def _initialize_model(self, model_type: str) -> None:
        """Initialize the ML model.
        
        Args:
            model_type: Type of model to initialize.
        """
        if model_type == 'gradient_boosting':
            params = self.config.get('gradient_boosting', {})
            self.model = GradientBoostingRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.1),
                subsample=params.get('subsample', 0.8),
                random_state=params.get('random_state', 42)
            )
        elif model_type == 'random_forest':
            params = self.config.get('random_forest', {})
            self.model = RandomForestRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 10),
                min_samples_split=params.get('min_samples_split', 5),
                min_samples_leaf=params.get('min_samples_leaf', 2),
                random_state=params.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the satisfaction prediction model.
        
        Args:
            X: Feature matrix.
            y: Target values (satisfaction scores).
        """
        logger.info(f"Training satisfaction predictor on {len(X)} samples")
        
        # Store feature names
        self.feature_names = [col for col in X.columns if col not in ['employee_id', 'feedback_text']]
        
        # Prepare features
        X_features = X[self.feature_names]
        
        # Fit model
        self.model.fit(X_features, y)
        self.is_fitted = True
        
        logger.info("Satisfaction predictor training completed")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict satisfaction scores.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Predicted satisfaction scores.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_features = X[self.feature_names]
        predictions = self.model.predict(X_features)
        
        # Clip predictions to valid range
        predictions = np.clip(predictions, 1.0, 5.0)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict satisfaction probability distribution.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Probability distribution over satisfaction levels.
        """
        predictions = self.predict(X)
        
        # Convert continuous predictions to probability distribution
        proba = np.zeros((len(predictions), 3))  # 3 classes: low, medium, high
        
        for i, pred in enumerate(predictions):
            if pred < 2.5:
                proba[i, 0] = 1.0  # Low satisfaction
            elif pred < 3.5:
                proba[i, 1] = 1.0  # Medium satisfaction
            else:
                proba[i, 2] = 1.0  # High satisfaction
        
        return proba
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores.
        
        Returns:
            DataFrame with feature importance.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        importance = self.model.feature_importances_
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model.
        
        Args:
            filepath: Path to save the model.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model.
        
        Args:
            filepath: Path to the saved model.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.config = model_data['config']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")


class FairnessAnalyzer:
    """Analyze fairness and bias in satisfaction predictions."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize fairness analyzer.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        self.protected_attributes = self.config.get('protected_attributes', [
            'age_group', 'gender', 'department', 'role_level'
        ])
        self.metrics = self.config.get('metrics', {})
    
    def analyze_fairness(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        sensitive_features: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze fairness metrics.
        
        Args:
            y_true: True satisfaction scores.
            y_pred: Predicted satisfaction scores.
            sensitive_features: DataFrame with protected attributes.
            
        Returns:
            Dictionary with fairness analysis results.
        """
        results = {}
        
        # Convert continuous predictions to binary for fairness analysis
        y_pred_binary = (y_pred >= 3.0).astype(int)
        y_true_binary = (y_true >= 3.0).astype(int)
        
        for attr in self.protected_attributes:
            if attr not in sensitive_features.columns:
                continue
            
            attr_results = {}
            
            # Demographic parity
            if self.metrics.get('demographic_parity', {}).get('enabled', True):
                try:
                    dp_diff = demographic_parity_difference(
                        y_true_binary, y_pred_binary, 
                        sensitive_features=sensitive_features[attr]
                    )
                    attr_results['demographic_parity_difference'] = dp_diff
                except Exception as e:
                    logger.warning(f"Demographic parity calculation failed for {attr}: {e}")
                    attr_results['demographic_parity_difference'] = None
            
            # Equalized odds
            if self.metrics.get('equalized_odds', {}).get('enabled', True):
                try:
                    eo_diff = equalized_odds_difference(
                        y_true_binary, y_pred_binary,
                        sensitive_features=sensitive_features[attr]
                    )
                    attr_results['equalized_odds_difference'] = eo_diff
                except Exception as e:
                    logger.warning(f"Equalized odds calculation failed for {attr}: {e}")
                    attr_results['equalized_odds_difference'] = None
            
            # Group-wise statistics
            attr_results['group_stats'] = self._calculate_group_stats(
                y_true, y_pred, sensitive_features[attr]
            )
            
            results[attr] = attr_results
        
        return results
    
    def _calculate_group_stats(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        groups: pd.Series
    ) -> Dict[str, Any]:
        """Calculate statistics for each group.
        
        Args:
            y_true: True satisfaction scores.
            y_pred: Predicted satisfaction scores.
            groups: Group assignments.
            
        Returns:
            Dictionary with group statistics.
        """
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
        
        return group_stats


class ModelEvaluator:
    """Evaluate model performance and generate reports."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize model evaluator.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        self.fairness_analyzer = FairnessAnalyzer(config.get('fairness', {}))
    
    def evaluate_model(
        self,
        model: SatisfactionPredictor,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sensitive_features: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Evaluate model performance.
        
        Args:
            model: Trained satisfaction predictor.
            X_test: Test features.
            y_test: Test targets.
            sensitive_features: Protected attributes for fairness analysis.
            
        Returns:
            Dictionary with evaluation results.
        """
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Calculate regression metrics
        regression_metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
            'r2': model.model.score(X_test[model.feature_names], y_test)
        }
        
        # Calculate classification metrics (using 3.0 as threshold)
        y_test_binary = (y_test >= 3.0).astype(int)
        y_pred_binary = (y_pred >= 3.0).astype(int)
        
        classification_metrics = {
            'accuracy': accuracy_score(y_test_binary, y_pred_binary),
            'f1_score': f1_score(y_test_binary, y_pred_binary, average='weighted')
        }
        
        # Fairness analysis
        fairness_results = {}
        if sensitive_features is not None:
            fairness_results = self.fairness_analyzer.analyze_fairness(
                y_test.values, y_pred, sensitive_features
            )
        
        # Feature importance
        feature_importance = model.get_feature_importance()
        
        return {
            'regression_metrics': regression_metrics,
            'classification_metrics': classification_metrics,
            'fairness_results': fairness_results,
            'feature_importance': feature_importance,
            'predictions': y_pred,
            'true_values': y_test.values
        }
    
    def generate_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate evaluation report.
        
        Args:
            evaluation_results: Results from evaluate_model.
            
        Returns:
            Formatted evaluation report.
        """
        report = []
        report.append("=" * 50)
        report.append("EMPLOYEE SATISFACTION MODEL EVALUATION REPORT")
        report.append("=" * 50)
        
        # Regression metrics
        report.append("\nREGRESSION METRICS:")
        report.append("-" * 20)
        for metric, value in evaluation_results['regression_metrics'].items():
            report.append(f"{metric.upper()}: {value:.4f}")
        
        # Classification metrics
        report.append("\nCLASSIFICATION METRICS:")
        report.append("-" * 20)
        for metric, value in evaluation_results['classification_metrics'].items():
            report.append(f"{metric.upper()}: {value:.4f}")
        
        # Fairness results
        if evaluation_results['fairness_results']:
            report.append("\nFAIRNESS ANALYSIS:")
            report.append("-" * 20)
            for attr, results in evaluation_results['fairness_results'].items():
                report.append(f"\n{attr.upper()}:")
                for metric, value in results.items():
                    if metric != 'group_stats':
                        report.append(f"  {metric}: {value:.4f}" if value is not None else f"  {metric}: N/A")
        
        # Top features
        report.append("\nTOP 10 MOST IMPORTANT FEATURES:")
        report.append("-" * 30)
        top_features = evaluation_results['feature_importance'].head(10)
        for _, row in top_features.iterrows():
            report.append(f"{row['feature']}: {row['importance']:.4f}")
        
        report.append("\n" + "=" * 50)
        
        return "\n".join(report)
