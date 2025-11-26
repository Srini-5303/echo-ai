# Data-Pipeline/scripts/bias_detection.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_rating_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze bias in rating distribution"""
    rating_dist = df['reviewRating'].value_counts(normalize=True).sort_index()
    
    analysis = {
        'distribution': rating_dist.to_dict(),
        'mean_rating': df['reviewRating'].mean(),
        'median_rating': df['reviewRating'].median(),
        'std_rating': df['reviewRating'].std(),
        'skewness': df['reviewRating'].skew(),
        'positive_ratio': (df['rareviewRatingting'] >= 4).mean(),
        'negative_ratio': (df['reviewRating'] <= 2).mean(),
        'neutral_ratio': (df['reviewRating'] == 3).mean()
    }
    
    # Check for imbalance
    if analysis['positive_ratio'] > 0.7:
        analysis['bias_detected'] = 'Positive bias - over 70% positive reviews'
    elif analysis['negative_ratio'] > 0.5:
        analysis['bias_detected'] = 'Negative bias - over 50% negative reviews'
    else:
        analysis['bias_detected'] = 'No significant bias detected'
    
    return analysis

def analyze_categorical_bias(df: pd.DataFrame, category_col: str) -> Dict[str, Any]:
    """Analyze bias across categorical variables"""
    if category_col not in df.columns:
        return {}
    
    category_analysis = {}
    categories = df[category_col].unique()
    
    for category in categories:
        category_df = df[df[category_col] == category]
        category_analysis[str(category)] = {
            'count': len(category_df),
            'percentage': (len(category_df) / len(df)) * 100,
            'avg_rating': category_df['rating'].mean(),
            'rating_std': category_df['rating'].std()
        }
    
    # Check for category imbalance
    counts = [v['count'] for v in category_analysis.values()]
    max_count = max(counts)
    min_count = min(counts)
    
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    return {
        'categories': category_analysis,
        'imbalance_ratio': imbalance_ratio,
        'most_common': max(category_analysis.items(), key=lambda x: x[1]['count'])[0],
        'least_common': min(category_analysis.items(), key=lambda x: x[1]['count'])[0]
    }

def analyze_temporal_bias(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze bias over time"""
    if 'reviewDate' not in df.columns:
        return {}
    
    df['reviewDate'] = pd.to_datetime(df['reviewDate'])
    df['month'] = df['date'].dt.to_period('M')
    
    temporal_analysis = df.groupby('month').agg({
        'rating': ['mean', 'std', 'count']
    })
    
    return {
        'monthly_trend': temporal_analysis.to_dict(),
        'trend_description': 'Temporal analysis completed'
    }

def analyze_text_length_bias(df: pd.DataFrame) -> Dict[str, Any]:
    if "text_length" not in df.columns:
        return {}

    short = df[df["text_length"] < 20]
    long_reviews = df[df["text_length"] > df["text_length"].quantile(0.9)]

    return {
        "short_reviews_count": len(short),
        "long_reviews_count": len(long_reviews),
        "avg_rating_short": short["reviewRating"].mean() if len(short) else None,
        "avg_rating_long": long_reviews["reviewRating"].mean() if len(long_reviews) else None,
        "bias_hint": "Short reviews may indicate spam/low-effort content"
    }


def detect_and_report_bias(input_path: str) -> Dict[str, Any]:
    try:
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)

        results = {
            "rating_distribution": analyze_rating_distribution(df),
            "sentiment_bias": analyze_categorical_bias(df, "sentiment"),
            "restaurant_bias": analyze_categorical_bias(df, "placeName"),
            "text_length_bias": analyze_text_length_bias(df)
        }

        if "reviewDate" in df.columns:
            results["temporal_bias"] = analyze_temporal_bias(df)

        # Save report
        os.makedirs("docs", exist_ok=True)
        with open("docs/bias_report.md", "w") as f:
            f.write("# Bias Detection Report\n\n")
            for section, data in results.items():
                f.write(f"## {section}\n")
                f.write(str(data) + "\n\n")

        logger.info("Bias detection complete.")
        return results

    except Exception as e:
        logger.error(f"Bias detection failed: {e}")
        raise

if __name__ == "__main__":
    detect_and_report_bias()
