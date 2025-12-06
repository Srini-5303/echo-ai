"""
Complete ML Pipeline Runner for EchoAI
Orchestrates all ML components in sequence
"""
import logging
import sys
import os
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import all modules
from config import *
from data_loader import prepare_data_for_training
from model_training import SentimentModelTrainer
from hyperparameter_tuning import tune_best_model
from model_validation import ModelValidator
from model_bias_detection import BiasDetector
from sensitivity_analysis import SensitivityAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'ml_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MLPipeline:
    """Main ML Pipeline orchestrator"""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.model = None
        self.vectorizer = None
        
    def print_header(self, title):
        """Print formatted section header"""
        print("\n" + "="*70)
        print(f"  {title}")
        print("="*70)
    
    def run_data_preparation(self):
        """Step 1: Data preparation"""
        self.print_header("STEP 1: DATA PREPARATION")
        
        try:
            data_splits, stats = prepare_data_for_training()
            
            self.results['data_preparation'] = {
                'status': 'success',
                'train_size': len(data_splits['train'][0]),
                'val_size': len(data_splits['val'][0]),
                'test_size': len(data_splits['test'][0]),
                'stats': stats
            }
            
            logger.info(" Data preparation completed successfully")
            return data_splits
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            self.results['data_preparation'] = {'status': 'failed', 'error': str(e)}
            raise
    
    def run_model_training(self, data_splits):
        """Step 2: Model training"""
        self.print_header("STEP 2: MODEL TRAINING")
        
        try:
            trainer = SentimentModelTrainer()
            best_model, vectorizer, training_results = trainer.train_all_models(data_splits)
            
            self.model = best_model
            self.vectorizer = vectorizer
            
            self.results['model_training'] = {
                'status': 'success',
                'best_model': trainer.best_model_name,
                'best_score': float(trainer.best_score),
                'all_scores': {k: float(v['f1_score']) for k, v in training_results.items()}
            }
            
            logger.info(" Model training completed successfully")
            return best_model, vectorizer
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            self.results['model_training'] = {'status': 'failed', 'error': str(e)}
            raise
    
    def run_hyperparameter_tuning(self, run_tuning=True):
        """Step 3: Hyperparameter tuning (optional)"""
        if not run_tuning:
            logger.info("Skipping hyperparameter tuning (set run_tuning=True to enable)")
            return None
        
        self.print_header("STEP 3: HYPERPARAMETER TUNING")
        
        try:
            # Tune the best model type from training
            model_type = self.results.get('model_training', {}).get('best_model', 'LogisticRegression')
            tuned_model, best_params = tune_best_model(model_type)
            
            self.results['hyperparameter_tuning'] = {
                'status': 'success',
                'model_type': model_type,
                'best_params': best_params
            }
            
            # Update model if tuning improved performance
            self.model = tuned_model
            
            logger.info(" Hyperparameter tuning completed successfully")
            return tuned_model
            
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {e}")
            self.results['hyperparameter_tuning'] = {'status': 'failed', 'error': str(e)}
            # Continue with untuned model
            return None
    
    def run_model_validation(self, data_splits):
        """Step 4: Model validation"""
        self.print_header("STEP 4: MODEL VALIDATION")
        
        try:
            validator = ModelValidator(self.model, self.vectorizer)
            X_test, y_test, metadata = data_splits['test']
            
            validation_results = validator.validate_model(X_test, y_test, metadata)
            
            self.results['model_validation'] = {
                'status': 'success',
                'accuracy': validation_results['metrics']['accuracy'],
                'f1_score': validation_results['metrics']['weighted_f1'],
                'auc': validation_results['metrics'].get('weighted_auc', 'N/A')
            }
            
            logger.info(" Model validation completed successfully")
            return validation_results
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            self.results['model_validation'] = {'status': 'failed', 'error': str(e)}
            raise
    
    def run_bias_detection(self, data_splits):
        """Step 5: Bias detection"""
        self.print_header("STEP 5: BIAS DETECTION")
        
        try:
            detector = BiasDetector(self.model, self.vectorizer)
            X_test, y_test, metadata = data_splits['test']
            
            bias_report = detector.run_bias_detection(X_test, y_test, metadata)
            
            self.results['bias_detection'] = {
                'status': 'success',
                'violations_found': len(bias_report['violations']),
                'recommendations': len(bias_report['recommendations'])
            }
            
            logger.info(" Bias detection completed successfully")
            return bias_report
            
        except Exception as e:
            logger.error(f"Bias detection failed: {e}")
            self.results['bias_detection'] = {'status': 'failed', 'error': str(e)}
            # Continue pipeline even if bias detection fails
            return None
    
    def run_sensitivity_analysis(self):
        """Step 6: Sensitivity analysis"""
        self.print_header("STEP 6: SENSITIVITY ANALYSIS")
        
        try:
            analyzer = SensitivityAnalyzer(self.model, self.vectorizer)
            sensitivity_results = analyzer.run_complete_sensitivity_analysis()
            
            self.results['sensitivity_analysis'] = {
                'status': 'success',
                'analyses_performed': sensitivity_results['analyses_performed']
            }
            
            logger.info(" Sensitivity analysis completed successfully")
            return sensitivity_results
            
        except Exception as e:
            logger.error(f"Sensitivity analysis failed: {e}")
            self.results['sensitivity_analysis'] = {'status': 'failed', 'error': str(e)}
            # Continue pipeline even if sensitivity analysis fails
            return None
    
    def generate_final_report(self):
        """Generate final pipeline report"""
        import json
        from datetime import datetime
        
        elapsed_time = time.time() - self.start_time
        
        final_report = {
            'pipeline_name': 'EchoAI ML Pipeline',
            'execution_date': datetime.now().isoformat(),
            'total_duration_seconds': elapsed_time,
            'pipeline_status': 'completed',
            'steps_summary': self.results
        }
        
        # Save final report
        report_path = RESULTS_DIR / 'pipeline_final_report.json'
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        # Print summary
        self.print_header("PIPELINE EXECUTION SUMMARY")
        print(f"\n Pipeline completed in {elapsed_time:.2f} seconds")
        print(f"\n Results saved to: {RESULTS_DIR}")
        print("\n Steps Summary:")
        
        for step, result in self.results.items():
            status = result.get('status', 'unknown')
            icon = "✓" if status == 'success' else "✗"
            print(f"  {icon} {step.replace('_', ' ').title()}: {status}")
        
        if self.model:
            print(f"\n Best Model: {self.results.get('model_training', {}).get('best_model', 'Unknown')}")
            print(f" Test F1 Score: {self.results.get('model_validation', {}).get('f1_score', 'N/A'):.4f}")
        
        print("\n" + "="*70)
        
        return final_report
    
    def run_pipeline(self, run_tuning=False):
        """Run complete ML pipeline"""
        self.start_time = time.time()
        
        print("     ECHO AI ML PIPELINE - STARTING EXECUTION")
        
        try:
            # Step 1: Data Preparation
            data_splits = self.run_data_preparation()
            
            # Step 2: Model Training
            model, vectorizer = self.run_model_training(data_splits)
            
            # Step 3: Hyperparameter Tuning (optional)
            if run_tuning:
                tuned_model = self.run_hyperparameter_tuning(run_tuning)
                if tuned_model:
                    model = tuned_model
            
            # Step 4: Model Validation
            validation_results = self.run_model_validation(data_splits)
            
            # Step 5: Bias Detection
            bias_report = self.run_bias_detection(data_splits)
            
            # Step 6: Sensitivity Analysis
            sensitivity_results = self.run_sensitivity_analysis()
            
            # Generate Final Report
            final_report = self.generate_final_report()
            
            logger.info(" Pipeline execution completed successfully!")
            
            return {
                'model': self.model,
                'vectorizer': self.vectorizer,
                'final_report': final_report
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            self.results['pipeline_status'] = 'failed'
            self.generate_final_report()
            raise

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run EchoAI ML Pipeline')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    parser.add_argument('--quick', action='store_true', help='Run quick version (skip optional steps)')
    args = parser.parse_args()
    
    # Create pipeline instance
    pipeline = MLPipeline()
    
    # Run pipeline
    try:
        if args.quick:
            logger.info("Running in quick mode (skipping hyperparameter tuning)")
            results = pipeline.run_pipeline(run_tuning=False)
        else:
            results = pipeline.run_pipeline(run_tuning=args.tune)
        
        print("\n EchoAI ML Pipeline completed successfully!")
        print(f" All results saved to: {RESULTS_DIR}")
        print(f" MLflow tracking UI: Run 'mlflow ui' to view experiment tracking")
        
    except Exception as e:
        print(f"\n Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()