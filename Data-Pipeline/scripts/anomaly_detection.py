# Data-Pipeline/scripts/anomaly_detection.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any
from scipy import stats
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect exact and text-based duplicates"""

    duplicates = {}

    # Exact duplicates
    duplicates["exact_duplicates"] = df[df.duplicated(keep=False)]

    # Text duplicates if reviewText exists
    if "reviewText" in df.columns:
        duplicates["duplicate_texts"] = df[df.duplicated(subset=['reviewText'], keep=False)]
    else:
        duplicates["duplicate_texts"] = pd.DataFrame()

    summary = {
        "exact_duplicate_count": len(duplicates["exact_duplicates"]),
        "duplicate_text_count": len(duplicates["duplicate_texts"]),
        "duplicate_indices": {
            "exact": duplicates["exact_duplicates"].index.tolist()[:10],
            "text": duplicates["duplicate_texts"].index.tolist()[:10],
        }
    }

    return summary


def detect_suspicious_patterns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    patterns = []

    # Very short reviews
    if "text_length" in df.columns:
        short_reviews = df[df["text_length"] < 10]

        if len(short_reviews) > 0:
            patterns.append({
                "type": "very_short_reviews",
                "count": len(short_reviews),
                "indices": short_reviews.index.tolist()[:10],
                "description": "Reviews with <10 characters"
            })

    # Repeated template-like reviews
    if "reviewText" in df.columns:
        template_like = df[df["reviewText"].str.len() < 50]
        counts = template_like["reviewText"].value_counts()

        common_templates = counts[counts > 5]  # Appears more than 5 times
        if len(common_templates) > 0:
            patterns.append({
                "type": "repeated_templates",
                "examples": common_templates.index.tolist()[:5],
                "description": "Short repeated template-like review texts"
            })

    return patterns

def detect_missing_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    missing_summary = {}

    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            missing_summary[col] = {
                "missing_count": missing,
                "missing_percentage": round((missing / len(df)) * 100, 2),
                "missing_indices": df[df[col].isnull()].index.tolist()[:10]
            }

    return missing_summary


def detect_numeric_outliers(df: pd.DataFrame) -> Dict[str, Any]:
    outlier_report = {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        col_z = stats.zscore(df[col].dropna())
        if len(col_z) == 0:
            continue

        outliers = df[col].iloc[np.where(abs(col_z) > 3)[0]]  # Z > 3

        if len(outliers) > 0:
            outlier_report[f"{col}_outliers"] = {
                "count": len(outliers),
                "indices": outliers.index.tolist()[:10],
                "description": f"Detected outliers in {col}"
            }

    return outlier_report


def detect_date_anomalies(df: pd.DataFrame) -> Dict[str, Any]:
    anomalies = {}

    if "reviewDate" not in df.columns:
        return anomalies

    try:
        df["reviewDate"] = pd.to_datetime(df["reviewDate"], errors="coerce")
    except:
        return {"date_format_errors": "Could not parse dates"}

    # Future dates
    future_dates = df[df["reviewDate"] > pd.Timestamp.today()]
    if len(future_dates) > 0:
        anomalies["future_dates"] = {
            "count": len(future_dates),
            "indices": future_dates.index.tolist()[:10]
        }

    # Very old dates (< 1990)
    old_dates = df[df["reviewDate"] < pd.Timestamp("1990-01-01")]
    if len(old_dates) > 0:
        anomalies["old_dates"] = {
            "count": len(old_dates),
            "indices": old_dates.index.tolist()[:10]
        }

    return anomalies

def detect_rating_anomalies(df: pd.DataFrame) -> Dict[str, Any]:
    anomalies = {}

    if "reviewRating" not in df.columns:
        return anomalies

    invalid = df[(df["reviewRating"] < 1) | (df["reviewRating"] > 5)]
    if len(invalid) > 0:
        anomalies["invalid_ratings"] = {
            "count": len(invalid),
            "indices": invalid.index.tolist()[:10]
        }

    # Class imbalance
    imbalance = df["reviewRating"].value_counts(normalize=True)

    if imbalance.max() > 0.8:
        anomalies["rating_imbalance"] = {
            "dominant_rating": imbalance.idxmax(),
            "percentage": round(imbalance.max() * 100, 2)
        }

    return anomalies


def detect_anomalies(input_path: str) -> Dict[str, Any]:
    try:
        logger.info(f"Loading data for anomaly detection: {input_path}")
        df = pd.read_csv(input_path)

        anomalies = {}

        anomalies["duplicates"] = detect_duplicates(df)
        anomalies["numeric_outliers"] = detect_numeric_outliers(df)
        anomalies["suspicious_patterns"] = detect_suspicious_patterns(df)
        anomalies["missing_patterns"] = detect_missing_patterns(df)
        anomalies["date_anomalies"] = detect_date_anomalies(df)
        anomalies["rating_anomalies"] = detect_rating_anomalies(df)

        logger.info("Anomaly detection completed.")
        return anomalies

    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise


if __name__ == "__main__":
    detect_anomalies("data/processed/clean_reviews.csv")
