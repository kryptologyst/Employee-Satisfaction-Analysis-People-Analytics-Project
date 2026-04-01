# Employee Satisfaction Analysis - People Analytics Project

## DISCLAIMER

**IMPORTANT: This is an experimental research and educational project. This tool is NOT intended for automated decision-making without human review. All outputs should be validated by qualified HR professionals and used only as supplementary information for human decision-making processes.**

## Overview

This project provides comprehensive employee satisfaction analysis using advanced NLP techniques, sentiment analysis, and fairness-aware machine learning models. It's designed for HR analytics, workforce insights, and organizational research purposes.

## Features

- **Advanced Sentiment Analysis**: Multiple NLP models including transformer-based approaches
- **Fairness-Aware Analytics**: Bias detection and mitigation for protected attributes
- **Comprehensive Evaluation**: Business and ML metrics with explainability
- **Interactive Dashboard**: Streamlit-based demo for exploration
- **Privacy-First Design**: PII minimization and anonymization features

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Employee-Satisfaction-Analysis-People-Analytics-Project.git
cd Employee-Satisfaction-Analysis-People-Analytics-Project

# Install dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Download required NLP models
python -m spacy download en_core_web_sm
```

### Basic Usage

```python
from src.models.sentiment_analyzer import SentimentAnalyzer
from src.data.synthetic_data import generate_employee_data

# Generate synthetic data
data = generate_employee_data(n_samples=1000)

# Initialize analyzer
analyzer = SentimentAnalyzer()

# Analyze satisfaction
results = analyzer.analyze_satisfaction(data)
print(results.summary())
```

### Run Demo

```bash
streamlit run demo/app.py
```

## Dataset Schema

### Employee Data (`workforce.*`)
- `employee_id`: Unique identifier (anonymized)
- `department`: Department/team
- `role_level`: Seniority level
- `tenure_months`: Time at company
- `feedback_text`: Open-ended survey responses
- `satisfaction_score`: Numeric satisfaction rating (1-5)
- `demographics`: Anonymized demographic info (age_group, gender, etc.)

### Survey Responses (`survey.*`)
- `response_id`: Unique response identifier
- `employee_id`: Reference to employee
- `timestamp`: Response date
- `question_type`: Type of question (open_text, rating, etc.)
- `response_text`: Text responses
- `response_value`: Numeric responses

## Training and Evaluation

### Train Models

```bash
# Train sentiment analysis models
python scripts/train_sentiment.py --config configs/sentiment_config.yaml

# Train satisfaction prediction model
python scripts/train_satisfaction.py --config configs/satisfaction_config.yaml

# Run fairness analysis
python scripts/fairness_analysis.py --config configs/fairness_config.yaml
```

### Evaluation

```bash
# Run comprehensive evaluation
python scripts/evaluate.py --config configs/eval_config.yaml

# Generate evaluation report
python scripts/generate_report.py --output assets/evaluation_report.html
```

## Metrics and KPIs

### ML Metrics
- **Sentiment Analysis**: Accuracy, F1-Score, AUROC, AUPRC
- **Satisfaction Prediction**: MAE, RMSE, MAPE, Calibration Error
- **Fairness**: Demographic Parity, Equalized Odds, Calibration

### Business KPIs
- **Satisfaction Trends**: Overall satisfaction score trends
- **Department Analysis**: Satisfaction by department/role
- **Risk Identification**: Employees at risk of dissatisfaction
- **Actionable Insights**: Key themes and improvement areas

## Limitations

- **Synthetic Data**: Uses generated data for demonstration
- **Model Limitations**: NLP models may have inherent biases
- **Privacy Considerations**: Requires careful handling of real employee data
- **Human Review Required**: All insights should be validated by HR professionals

## Configuration

Configuration files are located in `configs/`:
- `sentiment_config.yaml`: Sentiment analysis model settings
- `satisfaction_config.yaml`: Satisfaction prediction settings
- `fairness_config.yaml`: Fairness analysis parameters
- `eval_config.yaml`: Evaluation metrics and thresholds

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests
4. Run pre-commit hooks
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
# Employee-Satisfaction-Analysis-People-Analytics-Project
