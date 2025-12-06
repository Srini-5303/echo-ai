"""
Sensitivity analysis and model interpretability using SHAP and LIME
Adapted for dataset with features:
placeName, placeAddress, provider, reviewText, reviewDate, reviewRating, authorName, text
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import logging
from config import *
from data_loader import prepare_data_for_training

# Import interpretability libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not installed. Install with: pip install shap")

try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not installed. Install with: pip install lime")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensitivityAnalyzer:
    def __init__(self, model=None, vectorizer=None):

        # Load default model + vectorizer
        if model is None:
            self.model = joblib.load(BEST_MODEL_PATH)
            self.vectorizer = joblib.load(VECTORIZER_PATH)
        else:
            self.model = model
            self.vectorizer = vectorizer

        # Sentiment labels for your dataset
        # rating <= 2 → negative, rating=3 → neutral, rating>=4 → positive
        self.class_names = ["terrible","negative", "neutral", "positive", "amazing"]

        # Feature name extraction
        self.feature_names = None
        if hasattr(self.vectorizer, "get_feature_names_out"):
            self.feature_names = self.vectorizer.get_feature_names_out()

    def analyze_feature_importance(self):

        importance_data = {}

        # For LogisticRegression / Linear models
        if hasattr(self.model, "coef_"):
            logger.info("Analyzing feature importance for linear model")

            # Multiclass classifier → coef_ shape = (3, n_features)
            importance = np.abs(self.model.coef_).mean(axis=0)

            # Per-class importance
            for i, class_name in enumerate(self.class_names):
                class_importance = self.model.coef_[i]
                importance_data[f"{class_name}_positive"] = self.get_top_features(class_importance, 20, positive=True)
                importance_data[f"{class_name}_negative"] = self.get_top_features(class_importance, 20, positive=False)

            # Overall importance
            importance_data["overall_top_features"] = self.get_top_features(importance, 30)

        # Tree-based models
        elif hasattr(self.model, "feature_importances_"):
            logger.info("Analyzing feature importance for tree-based model")
            importance = self.model.feature_importances_
            importance_data["overall_top_features"] = self.get_top_features(importance, 30)

        else:
            logger.warning("Model does not provide feature importance values")
            return None

        # Plot
        self.plot_feature_importance(importance_data)
        return importance_data

    def get_top_features(self, importance_scores, n=20, positive=True):
        if self.feature_names is None:
            return []

        if positive:
            indices = np.argsort(importance_scores)[-n:][::-1]
        else:
            indices = np.argsort(importance_scores)[:n]

        return [
            {
                "feature": self.feature_names[i],
                "importance": float(importance_scores[i])
            }
            for i in indices if i < len(self.feature_names)
        ]

    def plot_feature_importance(self, importance_data, save_path=None):
        if "overall_top_features" not in importance_data:
            return

        top_features = importance_data["overall_top_features"][:20]
        features = [f["feature"] for f in top_features]
        scores = [f["importance"] for f in top_features]

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.title("Top 20 Most Important Features")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        save_path = save_path or (RESULTS_DIR / "feature_importance.png")
        plt.savefig(save_path, dpi=300)
        plt.show()

 
    def shap_analysis(self, X_sample, n_samples=10):

        if not SHAP_AVAILABLE:
            logger.warning("SHAP is not installed.")
            return None

        logger.info("Running SHAP…")

        # Sampling for speed
        if len(X_sample) > n_samples:
            idx = np.random.choice(len(X_sample), n_samples, replace=False)
            X_sample = X_sample[idx]

        X_vec = self.vectorizer.transform(X_sample)

        # KernelExplainer works for all models
        explainer = shap.KernelExplainer(
            lambda x: self.model.predict_proba(x),
            X_vec
        )

        shap_values = explainer.shap_values(X_vec)

        class_idx = min(2, len(shap_values) - 1)

        X_plot = X_vec  # 

        # Plot SHAP summary for positive class (index=2)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values[class_idx],
            X_plot,
            feature_names=self.feature_names,
            show=False
        )
        plt.title("SHAP Summary Plot (Class {class_idx})")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "shap_summary.png", dpi=300)
        plt.show()

        return shap_values

    def lime_explanation(self, text_sample, n_features=10):

        if not LIME_AVAILABLE:
            logger.warning("LIME is not installed.")
            return None

        logger.info("Generating LIME explanation…")

        explainer = LimeTextExplainer(class_names=self.class_names)

        def predict_fn(texts):
            X = self.vectorizer.transform(texts)
            return self.model.predict_proba(X)

        explanation = explainer.explain_instance(
            text_sample,
            predict_fn,
            num_features=n_features
        )

        fig = explanation.as_pyplot_figure()
        fig.suptitle("LIME Explanation", fontsize=14)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "lime_explanation.png", dpi=300)
        plt.show()

        return explanation.as_list()

    def analyze_prediction_confidence(self, X_test, y_test):

        if not hasattr(self.model, "predict_proba"):
            logger.warning("Model does not provide probabilities.")
            return None

        logger.info("Analyzing prediction confidence…")

        X_vec = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_vec)
        y_proba = self.model.predict_proba(X_vec)

        max_conf = np.max(y_proba, axis=1)
        correct = (y_pred == y_test)

        analysis = {
            "overall_mean_conf": float(max_conf.mean()),
            "correct_mean_conf": float(max_conf[correct].mean()),
            "incorrect_mean_conf": float(max_conf[~correct].mean()) if np.any(~correct) else None,
            "low_conf_samples": int(np.sum(max_conf < 0.5)),
            "high_conf_samples": int(np.sum(max_conf > 0.9))
        }

        plt.figure(figsize=(10, 5))
        plt.hist(max_conf, bins=30, alpha=0.8)
        plt.title("Confidence Distribution")
        plt.xlabel("Confidence")
        plt.ylabel("Count")
        plt.grid(alpha=0.3)
        plt.savefig(RESULTS_DIR / "confidence_distribution.png", dpi=300)
        plt.show()

        return analysis

    # -------------------------------------------------------------------
    # HYPERPARAMETER SENSITIVITY
    # -------------------------------------------------------------------
    def hyperparameter_sensitivity(self, X_train, y_train, X_val, y_val, param_name, param_range):

        from sklearn.metrics import f1_score

        logger.info(f"Hyperparameter sensitivity for {param_name}")

        results = []
        base_params = self.model.get_params()
        model_class = type(self.model)

        for val in param_range:
            try:
                params = base_params.copy()
                params[param_name] = val
                temp_model = model_class(**params)

                X_tr = self.vectorizer.transform(X_train)
                X_vl = self.vectorizer.transform(X_val)

                temp_model.fit(X_tr, y_train)
                y_pred = temp_model.predict(X_vl)

                score = f1_score(y_val, y_pred, average="weighted")
                results.append({"value": val, "f1": score})

            except Exception as e:
                logger.warning(f"Skipping {param_name}={val}: {e}")

        # Plot sensitivity
        if results:
            df = pd.DataFrame(results)
            plt.plot(df["value"], df["f1"], "o-")
            plt.xlabel(param_name)
            plt.ylabel("F1 Score")
            plt.title(f"Sensitivity to {param_name}")
            plt.grid(True)
            plt.savefig(RESULTS_DIR / f"sensitivity_{param_name}.png", dpi=300)
            plt.show()

        return results

    # -------------------------------------------------------------------
    # RUN FULL PIPELINE
    # -------------------------------------------------------------------
    def run_complete_sensitivity_analysis(self):

        logger.info("Running full sensitivity analysis…")

        data_splits, _ = prepare_data_for_training()

        X_train, y_train, _ = data_splits["train"]
        X_val, y_val, _ = data_splits["val"]
        X_test, y_test, _ = data_splits["test"]

        results = {}

        # 1. Feature importance
        results["feature_importance"] = self.analyze_feature_importance()

        # 2. SHAP (global)
        if SHAP_AVAILABLE:
            results["shap"] = self.shap_analysis(X_test[:100])

        # 3. LIME (local)
        if LIME_AVAILABLE:
            results["lime"] = self.lime_explanation(X_test[0])

        # 4. Confidence
        results["confidence"] = self.analyze_prediction_confidence(X_test, y_test)

        # 5. Hyperparameter sensitivity
        if hasattr(self.model, "C"):
            results["C_sensitivity"] = self.hyperparameter_sensitivity(
                X_train[:500], y_train[:500],
                X_val[:200], y_val[:200],
                "C", [0.01, 0.1, 1, 10, 100]
            )

        return results


def main():
    analyzer = SensitivityAnalyzer()
    return analyzer.run_complete_sensitivity_analysis()


if __name__ == "__main__":
    main()
