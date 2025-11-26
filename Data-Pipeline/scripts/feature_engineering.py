# Data-Pipeline/scripts/feature_engineering.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def assign_sentiment(rating: int) -> str:
    """Map numeric reviewRatings into simple sentiment labels."""
    
    sentiment_map = {
        5: "excellent",
        4: "positive",
        3: "neutral",
        2: "negative",
        1: "terrible"
    }
    
    return sentiment_map.get(rating, "unknown")

def compute_restaurant_avg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average reviewRatings for each restaurant (placeName).
    Adds a new column: restaurant_avg_rating
    """

    if "placeName" not in df.columns:
        raise ValueError("Missing required column: placeName")

    if "reviewRating" not in df.columns:
        raise ValueError("Missing required column: reviewRating")

    # Use groupby → hash map under the hood for speed
    avg_map = df.groupby("placeName")["reviewRating"].mean().to_dict()

    # Assign back to dataframe
    df["restaurant_avg_rating"] = df["placeName"].map(avg_map)

    return df

def create_features(
    input_path: str = "E:/Masters/MLOps/echo-ai/data/processed/clean_reviews_apify.csv",
    output_path: str = "E:/Masters/MLOps/echo-ai/data/processed/features_apify.csv"
) -> pd.DataFrame:

    try:
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)

        logger.info("Computing text length...")
        df["text_length"] = df["reviewText"].astype(str).apply(len)

        # Sentiment feature
        logger.info("Assigning sentiment categories...")
        df["sentiment"] = df["reviewRating"].apply(assign_sentiment)

        # Restaurant-level average rating
        logger.info("Computing restaurant average ratings...")
        df = compute_restaurant_avg(df)

        # Save
        df.to_csv(output_path, index=False)
        logger.info(
            f"Saved feature-enhanced data ({len(df)} rows, {len(df.columns)} columns) → {output_path}"
        )

        return df

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise


if __name__ == "__main__":
    create_features()

