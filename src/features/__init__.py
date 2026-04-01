"""Feature engineering module for employee satisfaction analysis.

This module provides feature extraction, transformation, and selection
capabilities for employee satisfaction data.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import LatentDirichletAllocation
import spacy
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)


class TextFeatureExtractor:
    """Extract features from text feedback."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize text feature extractor.
        
        Args:
            config: Configuration dictionary for feature extraction.
        """
        self.config = config or {}
        self.tfidf_vectorizer = None
        self.lda_model = None
        self.nlp = None
        
        # Initialize spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
        
        # Initialize VADER sentiment analyzer
        self.vader_analyzer = SentimentIntensityAnalyzer()
    
    def extract_sentiment_features(self, texts: List[str]) -> pd.DataFrame:
        """Extract sentiment features from text.
        
        Args:
            texts: List of text strings.
            
        Returns:
            DataFrame with sentiment features.
        """
        features = []
        
        for text in texts:
            if pd.isna(text) or text == "":
                features.append({
                    'textblob_polarity': 0.0,
                    'textblob_subjectivity': 0.0,
                    'vader_positive': 0.0,
                    'vader_negative': 0.0,
                    'vader_neutral': 1.0,
                    'vader_compound': 0.0
                })
                continue
            
            # TextBlob sentiment
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            
            # VADER sentiment
            vader_scores = self.vader_analyzer.polarity_scores(text)
            
            features.append({
                'textblob_polarity': textblob_polarity,
                'textblob_subjectivity': textblob_subjectivity,
                'vader_positive': vader_scores['pos'],
                'vader_negative': vader_scores['neg'],
                'vader_neutral': vader_scores['neu'],
                'vader_compound': vader_scores['compound']
            })
        
        return pd.DataFrame(features)
    
    def extract_linguistic_features(self, texts: List[str]) -> pd.DataFrame:
        """Extract linguistic features from text.
        
        Args:
            texts: List of text strings.
            
        Returns:
            DataFrame with linguistic features.
        """
        features = []
        
        for text in texts:
            if pd.isna(text) or text == "":
                features.append({
                    'text_length': 0,
                    'word_count': 0,
                    'sentence_count': 0,
                    'avg_word_length': 0.0,
                    'exclamation_count': 0,
                    'question_count': 0,
                    'capital_ratio': 0.0
                })
                continue
            
            # Basic text statistics
            text_length = len(text)
            words = text.split()
            word_count = len(words)
            sentences = text.split('.')
            sentence_count = len([s for s in sentences if s.strip()])
            
            avg_word_length = np.mean([len(word) for word in words]) if words else 0.0
            exclamation_count = text.count('!')
            question_count = text.count('?')
            capital_count = sum(1 for c in text if c.isupper())
            capital_ratio = capital_count / text_length if text_length > 0 else 0.0
            
            features.append({
                'text_length': text_length,
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_word_length': avg_word_length,
                'exclamation_count': exclamation_count,
                'question_count': question_count,
                'capital_ratio': capital_ratio
            })
        
        return pd.DataFrame(features)
    
    def extract_topic_features(self, texts: List[str], n_topics: int = 5) -> pd.DataFrame:
        """Extract topic features using LDA.
        
        Args:
            texts: List of text strings.
            n_topics: Number of topics to extract.
            
        Returns:
            DataFrame with topic features.
        """
        # Clean texts
        clean_texts = [str(text).lower() if pd.notna(text) else "" for text in texts]
        
        # Initialize TF-IDF vectorizer
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        
        # Fit TF-IDF
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(clean_texts)
        
        # Fit LDA model
        if self.lda_model is None:
            self.lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=100
            )
        
        topic_probs = self.lda_model.fit_transform(tfidf_matrix)
        
        # Create DataFrame with topic features
        topic_features = pd.DataFrame(
            topic_probs,
            columns=[f'topic_{i}' for i in range(n_topics)]
        )
        
        return topic_features
    
    def extract_named_entities(self, texts: List[str]) -> pd.DataFrame:
        """Extract named entities using spaCy.
        
        Args:
            texts: List of text strings.
            
        Returns:
            DataFrame with named entity features.
        """
        if self.nlp is None:
            logger.warning("spaCy model not available. Returning empty features.")
            return pd.DataFrame({'entity_count': [0] * len(texts)})
        
        features = []
        
        for text in texts:
            if pd.isna(text) or text == "":
                features.append({'entity_count': 0})
                continue
            
            doc = self.nlp(text)
            entity_count = len(doc.ents)
            
            features.append({'entity_count': entity_count})
        
        return pd.DataFrame(features)


class CategoricalFeatureEncoder:
    """Encode categorical features."""
    
    def __init__(self) -> None:
        """Initialize categorical feature encoder."""
        self.label_encoders = {}
        self.target_encoders = {}
    
    def fit_transform_categorical(
        self, 
        df: pd.DataFrame, 
        categorical_columns: List[str],
        target_column: Optional[str] = None,
        smoothing: float = 1.0
    ) -> pd.DataFrame:
        """Fit and transform categorical features.
        
        Args:
            df: Input DataFrame.
            categorical_columns: List of categorical column names.
            target_column: Target column for target encoding.
            smoothing: Smoothing parameter for target encoding.
            
        Returns:
            DataFrame with encoded categorical features.
        """
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col not in df.columns:
                continue
            
            # Label encoding
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            
            df_encoded[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                df[col].astype(str)
            )
            
            # Target encoding if target is provided
            if target_column and target_column in df.columns:
                if col not in self.target_encoders:
                    self.target_encoders[col] = {}
                
                # Calculate target means
                target_means = df.groupby(col)[target_column].mean()
                global_mean = df[target_column].mean()
                
                # Apply smoothing
                smoothed_means = (
                    (target_means * df.groupby(col).size() + global_mean * smoothing) /
                    (df.groupby(col).size() + smoothing)
                )
                
                self.target_encoders[col] = smoothed_means
                df_encoded[f'{col}_target_encoded'] = df[col].map(smoothed_means).fillna(global_mean)
        
        return df_encoded
    
    def transform_categorical(
        self, 
        df: pd.DataFrame, 
        categorical_columns: List[str],
        target_column: Optional[str] = None
    ) -> pd.DataFrame:
        """Transform categorical features using fitted encoders.
        
        Args:
            df: Input DataFrame.
            categorical_columns: List of categorical column names.
            target_column: Target column for target encoding.
            
        Returns:
            DataFrame with encoded categorical features.
        """
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col not in df.columns or col not in self.label_encoders:
                continue
            
            # Label encoding
            try:
                df_encoded[f'{col}_encoded'] = self.label_encoders[col].transform(
                    df[col].astype(str)
                )
            except ValueError:
                # Handle unseen categories
                df_encoded[f'{col}_encoded'] = 0
            
            # Target encoding
            if target_column and col in self.target_encoders:
                global_mean = df[target_column].mean() if target_column in df.columns else 0.0
                df_encoded[f'{col}_target_encoded'] = df[col].map(
                    self.target_encoders[col]
                ).fillna(global_mean)
        
        return df_encoded


class FeatureEngineer:
    """Main feature engineering class."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize feature engineer.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        self.text_extractor = TextFeatureExtractor(config.get('text_features', {}))
        self.categorical_encoder = CategoricalFeatureEncoder()
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit_transform(self, df: pd.DataFrame, target_column: str = 'satisfaction_score') -> pd.DataFrame:
        """Fit feature engineering pipeline and transform data.
        
        Args:
            df: Input DataFrame.
            target_column: Target column name.
            
        Returns:
            Transformed DataFrame with engineered features.
        """
        logger.info("Starting feature engineering...")
        
        # Extract text features
        if 'feedback_text' in df.columns:
            logger.info("Extracting text features...")
            
            # Sentiment features
            sentiment_features = self.text_extractor.extract_sentiment_features(
                df['feedback_text'].tolist()
            )
            
            # Linguistic features
            linguistic_features = self.text_extractor.extract_linguistic_features(
                df['feedback_text'].tolist()
            )
            
            # Topic features
            topic_features = self.text_extractor.extract_topic_features(
                df['feedback_text'].tolist()
            )
            
            # Named entity features
            entity_features = self.text_extractor.extract_named_entities(
                df['feedback_text'].tolist()
            )
            
            # Combine text features
            text_features = pd.concat([
                sentiment_features,
                linguistic_features,
                topic_features,
                entity_features
            ], axis=1)
            
            df = pd.concat([df, text_features], axis=1)
        
        # Encode categorical features
        categorical_columns = ['department', 'role_level', 'age_group', 'gender']
        df = self.categorical_encoder.fit_transform_categorical(
            df, categorical_columns, target_column
        )
        
        # Create derived features
        df = self._create_derived_features(df)
        
        # Scale numerical features
        numerical_columns = self._get_numerical_columns(df)
        if numerical_columns:
            df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])
        
        self.is_fitted = True
        logger.info(f"Feature engineering completed. Shape: {df.shape}")
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted pipeline.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            Transformed DataFrame.
        """
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted before transform")
        
        logger.info("Transforming data with fitted pipeline...")
        
        # Extract text features
        if 'feedback_text' in df.columns:
            # Sentiment features
            sentiment_features = self.text_extractor.extract_sentiment_features(
                df['feedback_text'].tolist()
            )
            
            # Linguistic features
            linguistic_features = self.text_extractor.extract_linguistic_features(
                df['feedback_text'].tolist()
            )
            
            # Topic features
            topic_features = self.text_extractor.extract_topic_features(
                df['feedback_text'].tolist()
            )
            
            # Named entity features
            entity_features = self.text_extractor.extract_named_entities(
                df['feedback_text'].tolist()
            )
            
            # Combine text features
            text_features = pd.concat([
                sentiment_features,
                linguistic_features,
                topic_features,
                entity_features
            ], axis=1)
            
            df = pd.concat([df, text_features], axis=1)
        
        # Encode categorical features
        categorical_columns = ['department', 'role_level', 'age_group', 'gender']
        df = self.categorical_encoder.transform_categorical(df, categorical_columns)
        
        # Create derived features
        df = self._create_derived_features(df)
        
        # Scale numerical features
        numerical_columns = self._get_numerical_columns(df)
        if numerical_columns:
            df[numerical_columns] = self.scaler.transform(df[numerical_columns])
        
        logger.info(f"Data transformation completed. Shape: {df.shape}")
        
        return df
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            DataFrame with derived features.
        """
        # Tenure-based features
        if 'tenure_months' in df.columns:
            df['tenure_years'] = df['tenure_months'] / 12
            df['is_new_employee'] = df['tenure_months'] < 12
            df['is_veteran'] = df['tenure_months'] > 60
        
        # Salary-based features
        if 'salary_percentile' in df.columns:
            df['is_high_paid'] = df['salary_percentile'] > 0.8
            df['is_low_paid'] = df['salary_percentile'] < 0.3
        
        # Interaction features
        if 'tenure_months' in df.columns and 'salary_percentile' in df.columns:
            df['tenure_salary_interaction'] = df['tenure_months'] * df['salary_percentile']
        
        return df
    
    def _get_numerical_columns(self, df: pd.DataFrame) -> List[str]:
        """Get numerical columns for scaling.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            List of numerical column names.
        """
        numerical_columns = []
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                # Skip target columns and encoded categorical columns
                if not any(skip in col.lower() for skip in ['target', 'encoded', 'id']):
                    numerical_columns.append(col)
        
        return numerical_columns
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names.
        
        Returns:
            List of feature names.
        """
        # This would be implemented based on the specific features created
        # For now, return a placeholder
        return []
