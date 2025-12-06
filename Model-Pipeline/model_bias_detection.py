"""
Bias detection module using data slicing techniques
Evaluates model fairness across different subgroups
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from config import *
from data_loader import prepare_data_for_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiasDetector:
    def __init__(self, model=None, vectorizer=None):
        if model is None:
            self.model = joblib.load(BEST_MODEL_PATH)
            self.vectorizer = joblib.load(VECTORIZER_PATH)
        else:
            self.model = model
            self.vectorizer = vectorizer
        
        self.metrics_to_check = ['accuracy', 'f1', 'precision', 'recall']
        self.bias_report = {}
    
    def evaluate_slice(self, X, y, slice_name=""):
        """Evaluate model performance on a data slice"""
        if len(X) < MIN_SLICE_SIZE:
            logger.warning(f"Slice '{slice_name}' has only {len(X)} samples (min: {MIN_SLICE_SIZE})")
            return None
        
        # Transform and predict
        X_tfidf = self.vectorizer.transform(X)
        y_pred = self.model.predict(X_tfidf)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'f1': f1_score(y, y_pred, average='weighted'),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted'),
            'support': len(y)
        }
        
        return metrics
    
    def detect_bias_by_slice(self, X, y, metadata, slice_column):
        """Detect bias by evaluating performance across different slices"""
        logger.info(f"Analyzing bias for: {slice_column}")
        
        slice_results = {}
        unique_values = metadata[slice_column].unique()
        
        # Evaluate each slice
        for value in unique_values:
            mask = metadata[slice_column] == value
            if mask.sum() > 0:
                X_slice = X[mask]
                y_slice = y[mask]
                
                metrics = self.evaluate_slice(X_slice, y_slice, f"{slice_column}={value}")
                if metrics:
                    slice_results[str(value)] = metrics
        
        # Calculate overall metrics
        X_tfidf = self.vectorizer.transform(X)
        y_pred = self.model.predict(X_tfidf)
        overall_metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'f1': f1_score(y, y_pred, average='weighted'),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted'),
            'support': len(y)
        }
        
        # Analyze disparities
        disparities = self.calculate_disparities(slice_results, overall_metrics)
        
        return {
            'slice_column': slice_column,
            'overall_metrics': overall_metrics,
            'slice_metrics': slice_results,
            'disparities': disparities
        }
    
    def calculate_disparities(self, slice_results, overall_metrics):
        """Calculate performance disparities across slices"""
        disparities = {}
        
        for metric in self.metrics_to_check:
            values = [s[metric] for s in slice_results.values() if s]
            if values:
                disparities[metric] = {
                    'max': max(values),
                    'min': min(values),
                    'range': max(values) - min(values),
                    'std': np.std(values),
                    'mean': np.mean(values),
                    'overall': overall_metrics[metric],
                    'max_deviation': max(abs(v - overall_metrics[metric]) for v in values)
                }
        
        return disparities
    
    def check_fairness_violations(self, disparities):
        """Check if fairness thresholds are violated"""
        violations = []
        
        for metric, stats in disparities.items():
            if stats['range'] > FAIRNESS_THRESHOLD:
                violations.append({
                    'metric': metric,
                    'range': stats['range'],
                    'threshold': FAIRNESS_THRESHOLD,
                    'severity': 'HIGH' if stats['range'] > FAIRNESS_THRESHOLD * 2 else 'MEDIUM'
                })
        
        return violations
    
    def create_bias_visualization(self, results_dict, save_path=None):
        """Create visualization of bias across slices"""
        n_slices = len(results_dict)
        fig, axes = plt.subplots(n_slices, 2, figsize=(15, 5*n_slices))
        if n_slices == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (slice_col, results) in enumerate(results_dict.items()):
            # Prepare data for plotting
            slice_names = list(results['slice_metrics'].keys())
            metrics_data = pd.DataFrame(results['slice_metrics']).T.dropna()

            available_metrics = [m for m in self.metrics_to_check if m in metrics_data.columns]

            if not available_metrics:
                logger.warning(f"No valid metric columns found for slice {slice_col}. Skipping plot.")
                ax1 = axes[idx, 0]
                ax1.text(0.5, 0.5, "No valid metrics", ha="center", va="center")
                ax1.set_title(f"{slice_col}: No Metrics Available")
                ax2 = axes[idx, 1]
                ax2.text(0.5, 0.5, "No support data", ha="center", va="center")
                ax2.set_title(f"{slice_col}: No Data")
                continue


            # Plot 1: Metrics comparison
            ax1 = axes[idx, 0]
            metrics_data[self.metrics_to_check].plot(kind='bar', ax=ax1)
            ax1.set_title(f'Performance Metrics by {slice_col}')
            ax1.set_xlabel(slice_col)
            ax1.set_ylabel('Score')
            ax1.legend(loc='best')
            ax1.grid(alpha=0.3)
            
            # Add overall performance line
            for metric in self.metrics_to_check:
                ax1.axhline(y=results['overall_metrics'][metric], 
                          linestyle='--', alpha=0.5, label=f'Overall {metric}')
            
            # Plot 2: Sample distribution
            ax2 = axes[idx, 1]
            if "support" in metrics_data.columns:
                support_data = metrics_data["support"].values
                ax2.bar(slice_names, support_data)
                ax2.set_ylabel("Number of Samples")
            else:
                ax2.text(0.5, 0.5, "No support column", ha="center", va="center")

            ax2.set_title(f'Sample Distribution across {slice_col}')
            ax2.set_xlabel(slice_col)
            ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Bias visualization saved to {save_path}")
        plt.show()
    
    def generate_bias_report(self, all_results):
        """Generate comprehensive bias report"""
        report = {
            'summary': {
                'total_slices_analyzed': len(all_results),
                'fairness_threshold': FAIRNESS_THRESHOLD,
                'min_slice_size': MIN_SLICE_SIZE
            },
            'slice_analyses': {},
            'violations': [],
            'recommendations': []
        }
        
        for slice_col, results in all_results.items():
            # Check for violations
            violations = self.check_fairness_violations(results['disparities'])
            
            report['slice_analyses'][slice_col] = {
                'metrics': results['slice_metrics'],
                'disparities': results['disparities'],
                'violations': violations
            }
            
            if violations:
                report['violations'].extend([{**v, 'slice': slice_col} for v in violations])
        
        # Generate recommendations
        if report['violations']:
            report['recommendations'] = self.generate_recommendations(report['violations'])
        
        return report
    
    def generate_recommendations(self, violations):
        """Generate bias mitigation recommendations"""
        recommendations = []
        
        # Group violations by severity
        high_severity = [v for v in violations if v.get('severity') == 'HIGH']
        medium_severity = [v for v in violations if v.get('severity') == 'MEDIUM']
        
        if high_severity:
            recommendations.append({
                'priority': 'HIGH',
                'issue': f"Found {len(high_severity)} high-severity fairness violations",
                'suggestion': "Consider re-training with balanced sampling or applying fairness constraints"
            })
        
        if medium_severity:
            recommendations.append({
                'priority': 'MEDIUM',
                'issue': f"Found {len(medium_severity)} medium-severity fairness violations",
                'suggestion': "Monitor these slices closely and consider targeted improvements"
            })
        
        # Specific recommendations based on violation patterns
        affected_slices = list(set(v['slice'] for v in violations))
        for slice_col in affected_slices:
            recommendations.append({
                'priority': 'MEDIUM',
                'issue': f"Performance disparity in {slice_col}",
                'suggestion': f"Apply data augmentation or re-weighting for underperforming {slice_col} categories"
            })
        
        return recommendations
    
    def run_bias_detection(self, X_test, y_test, metadata):
        """Run complete bias detection pipeline"""
        logger.info("="*60)
        logger.info("Starting Bias Detection Analysis")
        logger.info("="*60)
        
        all_results = {}
        
        # Analyze each slice dimension
        for slice_column in BIAS_SLICES:
            if slice_column in metadata.columns:
                results = self.detect_bias_by_slice(X_test, y_test, metadata, slice_column)
                all_results[slice_column] = results
            else:
                logger.warning(f"Slice column '{slice_column}' not found in metadata")
        
        # Create visualizations
        self.create_bias_visualization(
            all_results,
            save_path=RESULTS_DIR / 'bias_analysis.png'
        )
        
        # Generate report
        bias_report = self.generate_bias_report(all_results)
        
        # Save report
        import json
        with open(RESULTS_DIR / 'bias_report.json', 'w') as f:
            json.dump(bias_report, f, indent=2)
        
        # Print summary
        self.print_bias_summary(bias_report)
        
        return bias_report
    
    def print_bias_summary(self, report):
        """Print formatted bias detection summary"""
        print("\n" + "="*60)
        print("BIAS DETECTION SUMMARY")
        print("="*60)
        
        print(f"\nSlices Analyzed: {report['summary']['total_slices_analyzed']}")
        print(f"Fairness Threshold: {report['summary']['fairness_threshold']}")
        
        if report['violations']:
            print(f"\n  Found {len(report['violations'])} fairness violations:")
            for v in report['violations']:
                print(f"  - {v['slice']}/{v['metric']}: range={v['range']:.4f} (severity: {v['severity']})")
        else:
            print("\n No fairness violations detected!")
        
        if report['recommendations']:
            print("\n Recommendations:")
            for rec in report['recommendations']:
                print(f"  [{rec['priority']}] {rec['issue']}")
                print(f"    → {rec['suggestion']}")
        
        print("\n" + "="*60)

def main():
    """Main bias detection pipeline"""
    # Load test data
    data_splits, _ = prepare_data_for_training()
    X_test, y_test, metadata = data_splits['test']
    
    # Initialize bias detector
    detector = BiasDetector()
    

    # Run bias detection
    bias_report = detector.run_bias_detection(X_test, y_test, metadata)
    
    logger.info(f"Bias analysis complete. Report saved to {RESULTS_DIR}")
    
    return bias_report

if __name__ == "__main__":
    main()