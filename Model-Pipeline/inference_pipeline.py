"""
Complete Inference Pipeline for EchoAI
Combines sentiment analysis and response generation
"""
import joblib
import json
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime
import pandas as pd
from pathlib import Path

from config import *
from response_generator import ResponseGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EchoAIInference:
    """
    Complete inference pipeline for review analysis and response generation
    """
    
    def __init__(self, 
                 sentiment_model_path: Path = None,
                 vectorizer_path: Path = None,
                 llm_model: str = 'google/flan-t5-base'):
        """
        Initialize the inference pipeline
        
        Args:
            sentiment_model_path: Path to trained sentiment model
            vectorizer_path: Path to TF-IDF vectorizer
            llm_model: Name of the LLM model to use
        """
        self.sentiment_model_path = sentiment_model_path or BEST_MODEL_PATH
        self.vectorizer_path = vectorizer_path or VECTORIZER_PATH
        self.llm_model_name = llm_model
        
        self.sentiment_model = None
        self.vectorizer = None
        self.response_generator = None
        
        # Updated sentiment labels to match your 5 categories
        self.sentiment_labels = ['terrible', 'negative', 'neutral', 'positive', 'amazing']
        
        # Track performance metrics
        self.inference_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'avg_confidence': 0
        }
    
    def load_models(self, load_llm: bool = True):
        """
        Load all required models
        
        Args:
            load_llm: Whether to load the LLM (can be skipped for sentiment-only)
        """
        logger.info("Loading models for inference...")
        
        # Load sentiment analysis model
        try:
            self.sentiment_model = joblib.load(self.sentiment_model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            logger.info("✓ Sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            raise
        
        # Load LLM for response generation
        if load_llm:
            try:
                self.response_generator = ResponseGenerator(self.llm_model_name)
                self.response_generator.load_model()
                logger.info("✓ Response generation model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load LLM: {e}")
                logger.warning("Continuing without response generation")
                self.response_generator = None
    
    def predict_sentiment(self, text: str) -> Dict:
        """
        Predict sentiment for a single review
        
        Args:
            text: Review text
            
        Returns:
            Dictionary with sentiment prediction and confidence
        """
        if not self.sentiment_model or not self.vectorizer:
            raise ValueError("Sentiment model not loaded. Call load_models() first.")
        
        try:
            # Vectorize the text
            text_tfidf = self.vectorizer.transform([text])
            
            # Get prediction
            prediction = self.sentiment_model.predict(text_tfidf)[0]
            sentiment_label = self.sentiment_labels[prediction]
            
            # Get confidence if available
            confidence = None
            if hasattr(self.sentiment_model, 'predict_proba'):
                probabilities = self.sentiment_model.predict_proba(text_tfidf)[0]
                confidence = float(max(probabilities))
                
                # Get probability for each class
                class_probabilities = {
                    label: float(prob) 
                    for label, prob in zip(self.sentiment_labels, probabilities)
                }
            else:
                class_probabilities = {}
            
            return {
                'sentiment': sentiment_label,
                'sentiment_score': int(prediction),
                'confidence': confidence,
                'probabilities': class_probabilities
            }
            
        except Exception as e:
            logger.error(f"Error predicting sentiment: {e}")
            raise
    
    def generate_response(self, 
                         reviewText: str, 
                         sentiment: str = None,
                         placeName: str = None,
                         placeAddress: str = None,
                         provider: str = None,
                         reviewRating: float = None,
                         authorName: str = None,
                         reviewDate: str = None,
                         auto_detect_sentiment: bool = True) -> str:
        """
        Generate a response for a review using your specific features
        
        Args:
            reviewText: Review text
            sentiment: Sentiment (if None, will be predicted)
            placeName: Name of the place
            placeAddress: Address of the place
            provider: Review platform
            reviewRating: Customer rating
            authorName: Reviewer name
            reviewDate: Date of review
            auto_detect_sentiment: Whether to predict sentiment if not provided
            
        Returns:
            Generated response text
        """
        # Predict sentiment if not provided
        if sentiment is None and auto_detect_sentiment:
            sentiment_result = self.predict_sentiment(reviewText)
            sentiment = sentiment_result['sentiment']
        
        if not self.response_generator:
            logger.warning("Response generator not available")
            return self._get_template_response(sentiment)
        
        try:
            response = self.response_generator.generate_response(
                reviewText=reviewText, 
                sentiment=sentiment,
                placeName=placeName,
                placeAddress=placeAddress,
                provider=provider,
                reviewRating=reviewRating,
                authorName=authorName,
                reviewDate=reviewDate
            )
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._get_template_response(sentiment)
    
    def _get_template_response(self, sentiment: str) -> str:
        """Fallback template responses for 5 sentiment levels"""
        templates = {
            'amazing': "We are absolutely thrilled by your amazing review! Your incredible feedback means everything to us, and we can't wait to exceed your expectations again.",
            'positive': "Thank you for your positive feedback! We're delighted to hear about your experience and look forward to serving you again.",
            'neutral': "Thank you for taking the time to share your feedback. We value your input and are always working to improve our service.",
            'negative': "We sincerely apologize for your experience. Your feedback is important to us, and we'd like to make things right. Please contact us directly.",
            'terrible': "We are deeply sorry for the completely unacceptable experience you had. Please contact our management immediately so we can resolve this urgently."
        }
        return templates.get(sentiment, templates['neutral'])
    
    def process_review(self, 
                       review: Union[str, Dict],
                       generate_response: bool = True) -> Dict:
        """
        Process a single review through the complete pipeline
        
        Args:
            review: Review text or dict with review data
            generate_response: Whether to generate a response
            
        Returns:
            Complete analysis results
        """
        # Parse input based on your features
        if isinstance(review, str):
            review_text = review
            metadata = {}
        else:
            # Extract using your specific feature names
            review_text = review.get('reviewText', review.get('text', ''))
            metadata = {
                'placeName': review.get('placeName'),
                'placeAddress': review.get('placeAddress'),
                'provider': review.get('provider'),
                'reviewRating': review.get('reviewRating'),
                'authorName': review.get('authorName'),
                'reviewDate': review.get('reviewDate')
            }
            # Remove None values
            metadata = {k: v for k, v in metadata.items() if v is not None}
        
        # Start processing
        result = {
            'input': review_text,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Step 1: Sentiment Analysis
            sentiment_result = self.predict_sentiment(review_text)
            result['sentiment_analysis'] = sentiment_result
            
            # Step 2: Response Generation (if requested)
            if generate_response:
                response = self.generate_response(
                    reviewText=review_text,
                    sentiment=sentiment_result['sentiment'],
                    placeName=metadata.get('placeName'),
                    placeAddress=metadata.get('placeAddress'),
                    provider=metadata.get('provider'),
                    reviewRating=metadata.get('reviewRating'),
                    authorName=metadata.get('authorName'),
                    reviewDate=metadata.get('reviewDate')
                )
                result['generated_response'] = response
            
            # Update stats
            self.inference_stats['total_processed'] += 1
            self.inference_stats['successful'] += 1
            if sentiment_result.get('confidence'):
                self.inference_stats['avg_confidence'] = (
                    (self.inference_stats['avg_confidence'] * 
                     (self.inference_stats['successful'] - 1) +
                     sentiment_result['confidence']) / 
                    self.inference_stats['successful']
                )
            
            result['status'] = 'success'
            
        except Exception as e:
            logger.error(f"Error processing review: {e}")
            result['status'] = 'failed'
            result['error'] = str(e)
            self.inference_stats['failed'] += 1
        
        return result
    
    def process_batch(self, 
                     reviews: List[Union[str, Dict]],
                     generate_responses: bool = True,
                     save_results: bool = True) -> List[Dict]:
        """
        Process multiple reviews in batch
        
        Args:
            reviews: List of reviews (text or dicts)
            generate_responses: Whether to generate responses
            save_results: Whether to save results to file
            
        Returns:
            List of processed results
        """
        logger.info(f"Processing batch of {len(reviews)} reviews...")
        
        results = []
        for i, review in enumerate(reviews, 1):
            if i % 10 == 0:
                logger.info(f"Processing review {i}/{len(reviews)}")
            
            result = self.process_review(review, generate_responses)
            results.append(result)
        
        # Save results if requested
        if save_results:
            self._save_batch_results(results)
        
        # Print summary
        self._print_batch_summary(results)
        
        return results
    
    def process_dataframe(self, df: pd.DataFrame, 
                         generate_responses: bool = True,
                         save_results: bool = True) -> pd.DataFrame:
        """
        Process reviews from a DataFrame with your specific columns
        
        Args:
            df: DataFrame with review data
            generate_responses: Whether to generate responses
            save_results: Whether to save results
            
        Returns:
            DataFrame with added sentiment and response columns
        """
        # Convert DataFrame to list of dicts for processing
        reviews = df.to_dict('records')
        
        # Process all reviews
        results = self.process_batch(reviews, generate_responses, save_results=False)
        
        # Add results back to DataFrame
        df['sentiment'] = [r['sentiment_analysis']['sentiment'] for r in results]
        df['sentiment_confidence'] = [r['sentiment_analysis'].get('confidence', 0) for r in results]
        
        if generate_responses:
            df['generated_response'] = [r.get('generated_response', '') for r in results]
        
        # Save if requested
        if save_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = RESULTS_DIR / f'processed_reviews_{timestamp}.csv'
            df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
        
        return df
    
    def _save_batch_results(self, results: List[Dict]):
        """Save batch processing results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = RESULTS_DIR / f'inference_results_{timestamp}.json'
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    def _print_batch_summary(self, results: List[Dict]):
        """Print summary of batch processing"""
        successful = sum(1 for r in results if r.get('status') == 'success')
        failed = len(results) - successful
        
        sentiments = [
            r['sentiment_analysis']['sentiment'] 
            for r in results 
            if 'sentiment_analysis' in r
        ]
        
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        print(f"Total Processed: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
        if sentiments:
            print("\nSentiment Distribution:")
            for sentiment in self.sentiment_labels:
                count = sentiments.count(sentiment)
                percentage = (count / len(sentiments)) * 100 if sentiments else 0
                print(f"  {sentiment.capitalize():8} : {count:3} ({percentage:5.1f}%)")
        
        if self.inference_stats['avg_confidence'] > 0:
            print(f"\nAverage Confidence: {self.inference_stats['avg_confidence']:.3f}")
        
        print("="*60)
    
    def interactive_mode(self):
        """
        Interactive mode for testing individual reviews
        """
        print("     ECHOAI INTERACTIVE MODE")
        print("\nEnter reviews to analyze and generate responses.")
        print("Type 'quit' to exit, 'stats' for statistics.")
        print("-"*60)
        
        while True:
            try:
                # Get user input
                print("\n Enter a review (or command):")
                user_input = input("> ").strip()
                
                # Check for commands
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'stats':
                    print(f"\n Statistics:")
                    print(f"  Processed: {self.inference_stats['total_processed']}")
                    print(f"  Successful: {self.inference_stats['successful']}")
                    print(f"  Failed: {self.inference_stats['failed']}")
                    print(f"  Avg Confidence: {self.inference_stats['avg_confidence']:.3f}")
                    continue
                elif not user_input:
                    continue
                
                # Get optional metadata using your features
                print("\n Optional info (press Enter to skip):")
                place_name = input("  Place name: ").strip() or None
                place_address = input("  Place address: ").strip() or None
                provider = input("  Provider (Google/Yelp/TripAdvisor): ").strip() or None
                rating_str = input("  Rating (1-5): ").strip()
                rating = float(rating_str) if rating_str else None
                author = input("  Author name: ").strip() or None
                
                # Process review
                review_data = {
                    'reviewText': user_input,
                    'placeName': place_name,
                    'placeAddress': place_address,
                    'provider': provider,
                    'reviewRating': rating,
                    'authorName': author,
                    'reviewDate': datetime.now().strftime('%Y-%m-%d')
                }
                
                result = self.process_review(review_data, generate_response=True)
                
                # Display results
                print("\n" + "="*60)
                print("ANALYSIS RESULTS")
                print("="*60)
                
                if result['status'] == 'success':
                    sentiment_data = result['sentiment_analysis']
                    
                    print(f" Confidence: {sentiment_data.get('confidence', 0):.3f}")
                    
                    if sentiment_data.get('probabilities'):
                        print("\n Probabilities:")
                        for label in self.sentiment_labels:
                            prob = sentiment_data['probabilities'].get(label, 0)
                            print(f"  {label:8} {prob:.3f}")
                    
                    if 'generated_response' in result:
                        print(f"\n Generated Response:")
                        print(f"  {result['generated_response']}")
                else:
                    print(f" Error: {result.get('error', 'Unknown error')}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f" Error: {e}")
                continue

def main():
    """Main function to demonstrate the inference pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='EchoAI Inference Pipeline')
    parser.add_argument('--mode', choices=['interactive', 'batch', 'demo', 'csv'], 
                       default='demo', help='Running mode')
    parser.add_argument('--input', type=str, help='Input file for batch/csv mode')
    parser.add_argument('--llm', type=str, default='google/flan-t5-base',
                       help='LLM model to use')
    parser.add_argument('--no-response', action='store_true',
                       help='Skip response generation')
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = EchoAIInference(llm_model=args.llm)
    pipeline.load_models(load_llm=not args.no_response)
    
    if args.mode == 'interactive':
        # Interactive mode
        pipeline.interactive_mode()
        
    elif args.mode == 'batch':
        # Batch mode
        if not args.input:
            print("Error: --input required for batch mode")
            return
        
        # Load reviews from file
        if args.input.endswith('.json'):
            with open(args.input, 'r') as f:
                reviews = json.load(f)
        else:
            print("Error: Input file must be JSON for batch mode")
            return
        
        # Process batch
        results = pipeline.process_batch(
            reviews, 
            generate_responses=not args.no_response
        )
        
    elif args.mode == 'csv':
        # CSV mode for your specific features
        if not args.input:
            print("Error: --input required for CSV mode")
            return
        
        # Load CSV with your features
        df = pd.read_csv(args.input)
        
        # Check for required columns
        required_col = 'reviewText'
        if required_col not in df.columns:
            print(f"Error: CSV must contain '{required_col}' column")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Process DataFrame
        df_results = pipeline.process_dataframe(
            df,
            generate_responses=not args.no_response
        )
        
        print(f"\n✅ Processed {len(df_results)} reviews from CSV")
        
    else:
        # Demo mode with your features
        demo_reviews = [
            {
                'reviewText': "This place exceeded all my expectations! Absolutely phenomenal service and quality!",
                'placeName': 'The Grand Restaurant',
                'placeAddress': '123 Main St, Boston, MA',
                'provider': 'Google',
                'reviewRating': 5.0,
                'authorName': 'John Smith',
                'reviewDate': '2024-01-15'
            },
            {
                'reviewText': "Great food and excellent service. Would definitely recommend!",
                'placeName': 'Bella Italia',
                'placeAddress': '456 Oak Ave, Boston, MA',
                'provider': 'Yelp',
                'reviewRating': 4.0,
                'authorName': 'Sarah Johnson',
                'reviewDate': '2024-01-14'
            },
            {
                'reviewText': "Average place, nothing special. Service was okay.",
                'placeName': 'City Cafe',
                'placeAddress': '789 Park Rd, Boston, MA',
                'provider': 'TripAdvisor',
                'reviewRating': 3.0,
                'authorName': 'Mike Wilson',
                'reviewDate': '2024-01-13'
            },
            {
                'reviewText': "Very disappointed. Food was cold and service was slow.",
                'placeName': 'Quick Bites',
                'placeAddress': '321 Elm St, Boston, MA',
                'provider': 'Google',
                'reviewRating': 2.0,
                'authorName': 'Lisa Brown',
                'reviewDate': '2024-01-12'
            },
            {
                'reviewText': "Worst experience ever! Rude staff and terrible food. Never coming back!",
                'placeName': 'Corner Diner',
                'placeAddress': '654 Pine Ave, Boston, MA',
                'provider': 'Yelp',
                'reviewRating': 1.0,
                'authorName': 'Robert Davis',
                'reviewDate': '2024-01-11'
            }
        ]
        
        print("\n DEMO MODE - Processing sample reviews")
        print("="*60)
        
        for i, review in enumerate(demo_reviews, 1):
            print(f"\n Review {i}:")
            print(f"   Place: {review['placeName']}")
            print(f"   Review: {review['reviewText']}")
            print(f"   Author: {review['authorName']}")
            
            result = pipeline.process_review(review)
            
            if result['status'] == 'success':
                sentiment = result['sentiment_analysis']['sentiment']
                print(f" Confidence: {result['sentiment_analysis'].get('confidence', 0):.3f}")
                if 'generated_response' in result:
                    print(f" Response: {result['generated_response']}")
            print("-"*60)

if __name__ == "__main__":
    main()