"""
Configuration file for EchoAI ML Pipeline
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / 'data'
MODEL_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
LOGS_DIR = BASE_DIR / 'logs'

# Data paths
RAW_DATA_PATH = DATA_DIR / 'raw' / 'apify_data.csv'
PROCESSED_DATA_PATH = DATA_DIR / 'processed' / 'features_apify.csv'

# Model paths
BEST_MODEL_PATH = MODEL_DIR / 'best_model.pkl'
VECTORIZER_PATH = MODEL_DIR / 'tfidf_vectorizer.pkl'

# MLflow settings
MLFLOW_TRACKING_URI = 'mlruns'
MLFLOW_EXPERIMENT_NAME = 'echoai_sentiment_analysis'

# Model parameters
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42

# Text processing
MAX_FEATURES = 10000
MAX_LENGTH = 500
NGRAM_RANGE = (1, 3)

# Training parameters
HYPERPARAMETER_SEARCH_TRIALS = 50
CV_FOLDS = 5

# Bias detection slices
BIAS_SLICES = [
    'authorName',
    'provider',
    'reviewRating',
    'placeName'
]

# Fairness thresholds
FAIRNESS_THRESHOLD = 0.1  # Max acceptable difference in metrics across slices
MIN_SLICE_SIZE = 10  # Minimum samples in a slice for reliable metrics

# Create directories if they don't exist
for dir_path in [MODEL_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)