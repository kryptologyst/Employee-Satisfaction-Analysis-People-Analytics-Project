"""Data module for employee satisfaction analysis.

This module provides data loading, preprocessing, and synthetic data generation
for employee satisfaction analysis.
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandera import Column, DataFrameSchema, check_input, check_output
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmployeeDataGenerator:
    """Generate synthetic employee satisfaction data for demonstration purposes.
    
    This class creates realistic synthetic data that mimics real employee
    survey responses while ensuring privacy and avoiding PII.
    """
    
    def __init__(self, random_state: int = 42) -> None:
        """Initialize the data generator.
        
        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)
        
        # Define realistic employee attributes
        self.departments = [
            "Engineering", "Marketing", "Sales", "HR", "Finance", 
            "Operations", "Customer Support", "Product Management"
        ]
        
        self.role_levels = ["Junior", "Mid", "Senior", "Lead", "Manager", "Director"]
        
        self.age_groups = ["22-30", "31-40", "41-50", "51-60", "60+"]
        
        self.genders = ["Male", "Female", "Non-binary", "Prefer not to say"]
        
        # Satisfaction-related feedback templates
        self.positive_feedback = [
            "I love working here. The culture is amazing and supportive!",
            "Great team collaboration and work-life balance.",
            "Excellent growth opportunities and learning environment.",
            "Management is very supportive and understanding.",
            "The company values align with my personal values.",
            "Flexible work arrangements and great benefits.",
            "Challenging projects that help me grow professionally.",
            "Recognition and appreciation for good work.",
            "Clear communication and transparent leadership.",
            "Innovative projects and cutting-edge technology."
        ]
        
        self.negative_feedback = [
            "Management doesn't listen to employee concerns.",
            "Too much pressure and unrealistic deadlines.",
            "No recognition for hard work and dedication.",
            "Poor work-life balance and long hours.",
            "Limited growth opportunities and career advancement.",
            "Unclear communication from leadership.",
            "Competitive and toxic work environment.",
            "Inadequate compensation and benefits.",
            "Lack of support and resources for projects.",
            "Micromanagement and lack of autonomy."
        ]
        
        self.neutral_feedback = [
            "It's okay, but there's room for improvement.",
            "Some aspects are good, others need work.",
            "Mixed experiences depending on the team.",
            "Average work environment with some positives.",
            "Could be better but not terrible.",
            "Some good policies, some questionable ones.",
            "Depends on the project and team dynamics.",
            "Reasonable place to work overall.",
            "Some challenges but manageable.",
            "Standard corporate environment."
        ]

    def generate_employee_data(
        self, 
        n_samples: int = 1000,
        satisfaction_distribution: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """Generate synthetic employee satisfaction data.
        
        Args:
            n_samples: Number of employee records to generate.
            satisfaction_distribution: Distribution of satisfaction levels.
                Defaults to realistic distribution.
                
        Returns:
            DataFrame with employee data including demographics and feedback.
        """
        if satisfaction_distribution is None:
            satisfaction_distribution = {
                "positive": 0.4,
                "neutral": 0.35,
                "negative": 0.25
            }
        
        data = []
        
        for i in range(n_samples):
            # Generate demographic attributes
            department = random.choice(self.departments)
            role_level = random.choice(self.role_levels)
            age_group = random.choice(self.age_groups)
            gender = random.choice(self.genders)
            
            # Generate tenure (correlated with age and role)
            if role_level in ["Junior"]:
                tenure_months = random.randint(1, 24)
            elif role_level in ["Mid"]:
                tenure_months = random.randint(12, 60)
            elif role_level in ["Senior", "Lead"]:
                tenure_months = random.randint(36, 120)
            else:  # Manager, Director
                tenure_months = random.randint(60, 180)
            
            # Generate satisfaction level based on distribution
            satisfaction_level = np.random.choice(
                list(satisfaction_distribution.keys()),
                p=list(satisfaction_distribution.values())
            )
            
            # Generate feedback text based on satisfaction level
            if satisfaction_level == "positive":
                feedback_text = random.choice(self.positive_feedback)
                satisfaction_score = random.uniform(3.5, 5.0)
            elif satisfaction_level == "negative":
                feedback_text = random.choice(self.negative_feedback)
                satisfaction_score = random.uniform(1.0, 2.5)
            else:  # neutral
                feedback_text = random.choice(self.neutral_feedback)
                satisfaction_score = random.uniform(2.5, 3.5)
            
            # Add some noise and correlations
            if department == "Engineering" and role_level in ["Senior", "Lead"]:
                satisfaction_score += random.uniform(0.2, 0.5)
            elif department == "Customer Support":
                satisfaction_score -= random.uniform(0.1, 0.3)
            
            # Generate additional attributes
            salary_percentile = random.uniform(0.1, 0.9)
            if role_level in ["Manager", "Director"]:
                salary_percentile = random.uniform(0.6, 0.95)
            
            # Generate survey response timestamp (within last 6 months)
            days_ago = random.randint(1, 180)
            response_date = pd.Timestamp.now() - pd.Timedelta(days=days_ago)
            
            employee_record = {
                "employee_id": f"EMP_{i:06d}",
                "department": department,
                "role_level": role_level,
                "age_group": age_group,
                "gender": gender,
                "tenure_months": tenure_months,
                "salary_percentile": salary_percentile,
                "feedback_text": feedback_text,
                "satisfaction_score": round(satisfaction_score, 2),
                "satisfaction_level": satisfaction_level,
                "response_date": response_date,
                "survey_version": "2024_Q1"
            }
            
            data.append(employee_record)
        
        df = pd.DataFrame(data)
        
        # Add some additional derived features
        df["is_long_tenure"] = df["tenure_months"] > 36
        df["is_high_salary"] = df["salary_percentile"] > 0.7
        df["feedback_length"] = df["feedback_text"].str.len()
        
        logger.info(f"Generated {n_samples} employee records")
        return df

    def generate_survey_responses(
        self, 
        employee_df: pd.DataFrame,
        n_responses_per_employee: int = 3
    ) -> pd.DataFrame:
        """Generate additional survey responses for each employee.
        
        Args:
            employee_df: Base employee data.
            n_responses_per_employee: Number of additional responses per employee.
            
        Returns:
            DataFrame with survey response data.
        """
        responses = []
        
        for _, employee in employee_df.iterrows():
            for i in range(n_responses_per_employee):
                # Generate different types of questions
                question_types = [
                    "work_environment",
                    "management_support", 
                    "career_growth",
                    "compensation",
                    "work_life_balance"
                ]
                
                question_type = random.choice(question_types)
                
                # Generate rating responses (1-5 scale)
                base_rating = employee["satisfaction_score"]
                rating = max(1, min(5, base_rating + random.uniform(-0.5, 0.5)))
                
                # Generate text response based on rating
                if rating >= 4:
                    text_response = random.choice(self.positive_feedback)
                elif rating <= 2:
                    text_response = random.choice(self.negative_feedback)
                else:
                    text_response = random.choice(self.neutral_feedback)
                
                response = {
                    "response_id": f"RESP_{employee['employee_id']}_{i}",
                    "employee_id": employee["employee_id"],
                    "question_type": question_type,
                    "rating_response": round(rating, 1),
                    "text_response": text_response,
                    "response_date": employee["response_date"] + pd.Timedelta(days=random.randint(-30, 30))
                }
                
                responses.append(response)
        
        return pd.DataFrame(responses)


# Data validation schemas
employee_schema = DataFrameSchema({
    "employee_id": Column(str),
    "department": Column(str),
    "role_level": Column(str),
    "age_group": Column(str),
    "gender": Column(str),
    "tenure_months": Column(int, checks=[lambda x: x >= 0]),
    "salary_percentile": Column(float, checks=[lambda x: (x >= 0) & (x <= 1)]),
    "feedback_text": Column(str),
    "satisfaction_score": Column(float, checks=[lambda x: (x >= 1) & (x <= 5)]),
    "satisfaction_level": Column(str),
    "response_date": Column("datetime64[ns]"),
    "survey_version": Column(str)
})

survey_response_schema = DataFrameSchema({
    "response_id": Column(str),
    "employee_id": Column(str),
    "question_type": Column(str),
    "rating_response": Column(float, checks=[lambda x: (x >= 1) & (x <= 5)]),
    "text_response": Column(str),
    "response_date": Column("datetime64[ns]")
})


class DataProcessor:
    """Process and validate employee satisfaction data."""
    
    def __init__(self) -> None:
        """Initialize the data processor."""
        self.logger = logging.getLogger(__name__)
    
    @check_input(employee_schema)
    def validate_employee_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate employee data against schema.
        
        Args:
            df: Employee data DataFrame.
            
        Returns:
            Validated DataFrame.
        """
        self.logger.info(f"Validated {len(df)} employee records")
        return df
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text data for analysis.
        
        Args:
            text: Raw text input.
            
        Returns:
            Preprocessed text.
        """
        if pd.isna(text):
            return ""
        
        # Basic text cleaning
        text = str(text).strip()
        text = text.lower()
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        return text
    
    def create_train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        validation_size: float = 0.2,
        random_state: int = 42,
        stratify_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train/validation/test splits.
        
        Args:
            df: Input DataFrame.
            test_size: Proportion of data for test set.
            validation_size: Proportion of data for validation set.
            random_state: Random seed.
            stratify_column: Column to stratify on.
            
        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        stratify = df[stratify_column] if stratify_column else None
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        # Second split: train vs val
        if stratify_column:
            stratify = train_val_df[stratify_column]
        
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=validation_size / (1 - test_size),
            random_state=random_state,
            stratify=stratify
        )
        
        self.logger.info(f"Split data: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        
        return train_df, val_df, test_df


def generate_employee_data(n_samples: int = 1000) -> pd.DataFrame:
    """Convenience function to generate employee data.
    
    Args:
        n_samples: Number of samples to generate.
        
    Returns:
        Generated employee DataFrame.
    """
    generator = EmployeeDataGenerator()
    return generator.generate_employee_data(n_samples)
