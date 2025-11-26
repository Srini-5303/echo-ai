# Data-Pipeline/scripts/preprocessing.py
import pandas as pd
import numpy as np
import re
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean and normalize review text"""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters but keep spaces and basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()

def generate_template_review(rating: int) -> str:
    templates = {
        5: "Absolutely amazing experience! Highly recommended.",
        4: "A very good experience overall.",
        3: "An average, neutral experience.",
        2: "A below-average experience. Could be better.",
        1: "This was the worst experience I’ve had."
    }
    return templates.get(rating, "")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Missing values before handling: {df.isnull().sum().sum()}")

    if "reviewRating" in df.columns:
        df = df.dropna(subset=["reviewRating"])

        df["reviewRating"] = df["reviewRating"].astype(int)

        df["reviewText"] = df.apply(
            lambda row: generate_template_review(row["reviewRating"]) 
            if pd.isna(row["reviewText"]) or str(row["reviewText"]).strip() == "" 
            else row["reviewText"],
            axis=1
        )
    
    logger.info(f"Missing values after handling: {df.isnull().sum().sum()}")
    return df

def preprocess_data(input_path: str = 'E:/Masters/MLOps/echo-ai/data/raw/apify_data.csv',
                   output_path: str = 'E:/Masters/MLOps/echo-ai/data/processed/clean_reviews_apify.csv') -> pd.DataFrame:
    """Main preprocessing pipeline"""
    try:
        # Load data
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} reviews")
        
        expected_cols = {"placeName", "placeAddress", "reviewText","reviewDate", "reviewRating", "authorName"}
        if not expected_cols.issubset(df.columns):
            missing = expected_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Handle missing values
        df = handle_missing_values(df)
        
        df["reviewDate"] = pd.to_datetime(df["reviewDate"], errors="coerce").dt.date

        # Clean text
        logger.info("Cleaning text...")
        df['reviewText'] = df['reviewText'].apply(clean_text)
        
        # Save processed data
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} processed reviews to {output_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    preprocess_data()
