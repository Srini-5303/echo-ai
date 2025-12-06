"""
Model validation module for comprehensive evaluation
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from config import *
from data_loader import prepare_data_for_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelValidator:
    def __init__(self, model=None, vectorizer=None):
        # Load default saved artifacts
        if model is None:
            self.model = joblib.load(BEST_MODEL_PATH)
            self.vectorizer = joblib.load(VECTORIZER_PATH)
        else:
            self.model = model
            self.vectorizer = vectorizer
        
        # FIXED: Five sentiment classes
        self.class_names = ["Amazing", "Positive", "Neutral", "Negative", "Terrible"]
        self.num_classes = len(self.class_names)

    def calculate_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate comprehensive metrics"""
        metrics = {}

        # Base accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=range(self.num_classes), zero_division=0
        )

        for i, class_name in enumerate(self.class_names):
            metrics[f"{class_name}_precision"] = precision[i]
            metrics[f"{class_name}_recall"] = recall[i]
            metrics[f"{class_name}_f1"] = f1[i]
            metrics[f"{class_name}_support"] = int(support[i])

        # Weighted averages
        total_support = support.sum()
        metrics["weighted_precision"] = np.average(precision, weights=support)
        metrics["weighted_recall"] = np.average(recall, weights=support)
        metrics["weighted_f1"] = np.average(f1, weights=support)

        # Macro averages
        metrics["macro_precision"] = precision.mean()
        metrics["macro_recall"] = recall.mean()
        metrics["macro_f1"] = f1.mean()

        # AUC (only if probabilities exist)
        if y_proba is not None:
            try:
                y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))

                for i, class_name in enumerate(self.class_names):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                    metrics[f"{class_name}_auc"] = auc(fpr, tpr)

                metrics["macro_auc"] = np.mean(
                    [metrics[f"{cn}_auc"] for cn in self.class_names]
                )
                weights = support / total_support
                metrics["weighted_auc"] = np.sum(
                    [metrics[f"{cn}_auc"] * w for cn, w in zip(self.class_names, weights)]
                )
            except Exception as e:
                logger.warning(f"AUC calculation skipped: {e}")

        return metrics

    def create_confusion_matrix_plot(self, y_true, y_pred, save_path=None):
        """Create and save confusion matrix visualization"""
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Confusion matrix saved to {save_path}")

        plt.show()
        return cm

    def create_roc_curves(self, y_true, y_proba, save_path=None):
        """Create ROC curves for each class"""
        y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))

        plt.figure(figsize=(12, 8))

        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc_val:.3f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves (5-Class Sentiment)")
        plt.legend()

        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"ROC curves saved to {save_path}")

        plt.show()

    def analyze_errors(self, X, y_true, y_pred, n_samples=10):
        misclassified = np.where(y_true != y_pred)[0]
        if len(misclassified) == 0:
            logger.info("No misclassified samples.")
            return []

        sample_indices = np.random.choice(
            misclassified, min(n_samples, len(misclassified)), replace=False
        )

        errors = []
        for idx in sample_indices:
            errors.append({
                "text": X[idx][:200] + "...",
                "true_label": self.class_names[y_true[idx]],
                "predicted_label": self.class_names[y_pred[idx]]
            })
        return errors

    def validate_model(self, X_test, y_test, metadata=None):
        """Main validation pipeline"""
        logger.info("Validating model...")

        X_test_vec = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_vec)

        # Convert 1–5 → 0–4
        y_pred = y_pred - 1
        y_test = y_test - 1

        try:
            y_proba = self.model.predict_proba(X_test_vec)
        except:
            y_proba = None

        metrics = self.calculate_metrics(y_test, y_pred, y_proba)

        cm = self.create_confusion_matrix_plot(
            y_test, y_pred, save_path=RESULTS_DIR / "confusion_matrix.png"
        )

        if y_proba is not None:
            self.create_roc_curves(
                y_test, y_proba, save_path=RESULTS_DIR / "roc_curves.png"
            )

        report = classification_report(
            y_test, y_pred, labels=range(self.num_classes),
            target_names=self.class_names, output_dict=True
        )

        errors = self.analyze_errors(X_test, y_test, y_pred)

        results = {
            "metrics": metrics,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "errors": errors
        }

        import json
        with open(RESULTS_DIR / "validation_results.json", "w") as f:
            json.dump(results, f, indent=2)

        logger.info("Validation completed.")
        self.print_summary(metrics)

        return results

    def print_summary(self, metrics):
        print("\n==== MODEL VALIDATION RESULTS ====")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {metrics['weighted_f1']:.4f}")

        print("\nPer-Class Performance:")
        for c in self.class_names:
            print(f"\n{c}")
            print(f"  Precision: {metrics[f'{c}_precision']:.4f}")
            print(f"  Recall:    {metrics[f'{c}_recall']:.4f}")
            print(f"  F1:        {metrics[f'{c}_f1']:.4f}")
            print(f"  Support:   {metrics[f'{c}_support']}")

def main():
    logger.info("Loading test split...")
    data_splits, _ = prepare_data_for_training()
    X_test, y_test, metadata = data_splits["test"]

    validator = ModelValidator()
    validator.validate_model(X_test, y_test, metadata)

if __name__ == "__main__":
    main()
