"""
Data loader module for EchoAI ML Pipeline
Updated to match dataset:
placeName, placeAddress, provider, reviewText, reviewDate, reviewRating, authorName
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_processed_data():
    """Load data and validate required fields"""
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        logger.info(f"Loaded {len(df)} rows from {PROCESSED_DATA_PATH}")

        # Dataset-specific required columns
        required_columns = [
            'reviewText',            
            'reviewRating'   
        ]

        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Ensure rating is numeric
        df['reviewRating'] = pd.to_numeric(df['reviewRating'], errors='coerce')

        # Drop rows with missing rating or text
        df = df.dropna(subset=['reviewText', 'reviewRating']).reset_index(drop=True)

        df['text_length'] = df['reviewText'].astype(str).apply(len)

        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def create_train_val_test_split(df, text_col='reviewText', label_col='reviewRating'):
    """Split into train/val/test using numeric review ratings"""
    
    X = df[text_col].values
    y = df[label_col].values

    metadata_cols = [
        'placeName',
        'placeAddress',
        'provider',
        'reviewDate',
        'authorName'
    ]

    # Keep only metadata that actually exists in df
    metadata = df[metadata_cols].copy()

    # Split 1: train+val vs test
    X_temp, X_test, y_temp, y_test, meta_temp, meta_test = train_test_split(
        X, y, metadata,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Split 2: train vs val
    val_size_adjusted = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
        X_temp, y_temp, meta_temp,
        test_size=val_size_adjusted,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return {
        'train': (X_train, y_train, meta_train),
        'val': (X_val, y_val, meta_val),
        'test': (X_test, y_test, meta_test)
    }


def get_data_stats(df):
    """Basic dataset stats using available columns"""
    stats = {
        'total_reviews': len(df),
        'avg_rating': df['reviewRating'].mean(),
        'rating_distribution': df['reviewRating'].value_counts().to_dict(),
        'avg_text_length': df['text_length'].mean(),
        'provider_distribution': df['provider'].value_counts().to_dict() if 'provider' in df else None
    }
    return stats


def prepare_data_for_training():
    logger.info("Starting data preparation for training")

    # Load data
    df = load_processed_data()

    # Compute stats
    stats = get_data_stats(df)
    logger.info(f"Dataset statistics: {stats}")

    # Split data
    data_splits = create_train_val_test_split(df)

    # Save split metadata
    split_info = {
        'train_size': len(data_splits['train'][0]),
        'val_size': len(data_splits['val'][0]),
        'test_size': len(data_splits['test'][0]),
        'random_state': RANDOM_STATE
    }

    import json
    with open(RESULTS_DIR / 'data_splits.json', 'w') as f:
        json.dump(split_info, f, indent=2)

    logger.info("Data preparation complete")
    return data_splits, stats


if __name__ == "__main__":
    data_splits, stats = prepare_data_for_training()
    print("\nData preparation successful!")
    print(f"Train samples: {len(data_splits['train'][0])}")
    print(f"Validation samples: {len(data_splits['val'][0])}")
    print(f"Test samples: {len(data_splits['test'][0])}")
