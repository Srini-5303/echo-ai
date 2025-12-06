"""
Hyperparameter tuning module using Optuna
Updated to automatically select best model from results.json
"""
import json
import optuna
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score, accuracy_score
import mlflow
import joblib
import logging
from config import *
from data_loader import prepare_data_for_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperparameterTuner:
    def __init__(self, model_type='LogisticRegression'):
        self.model_type = model_type
        self.best_params = None
        self.best_score = -1
        self.study = None
        
    def get_search_space(self, trial):
        """Define hyperparameter search space for each model type"""
        if self.model_type == "LogisticRegression":
            penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
            solver = trial.suggest_categorical("solver", ["liblinear", "saga"])
            
            if penalty == "l1" and solver == "saga":
                pass
            elif penalty == "l1" and solver != "liblinear":
                raise optuna.exceptions.TrialPruned()
            
            return {
                "C": trial.suggest_float("C", 0.001, 50, log=True),
                "penalty": penalty,
                "solver": solver,
                "max_iter": trial.suggest_int("max_iter", 200, 1500),
                "class_weight": "balanced"
            }
            
        elif self.model_type == "RandomForest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 8, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"])
            }
            
        elif self.model_type == "GradientBoosting":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 80, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0)
            }
            
        elif self.model_type == "LinearSVC":
            return {
                "C": trial.suggest_float("C", 0.001, 10, log=True),
                "max_iter": trial.suggest_int("max_iter", 1000, 3000),
                "class_weight": "balanced"
            }
            
        elif self.model_type == "MultinomialNB":
            return {
                "alpha": trial.suggest_float("alpha", 0.01, 1.0, log=True)
            }
            
    def create_model(self, params):
        if self.model_type == "LogisticRegression":
            return LogisticRegression(random_state=RANDOM_STATE, **params)
        elif self.model_type == "RandomForest":
            return RandomForestClassifier(random_state=RANDOM_STATE, **params)
        elif self.model_type == "GradientBoosting":
            return GradientBoostingClassifier(random_state=RANDOM_STATE, **params)
        elif self.model_type == "LinearSVC":
            return LinearSVC(random_state=RANDOM_STATE, **params)
        elif self.model_type == "MultinomialNB":
            return MultinomialNB(**params)
            
    def objective(self, trial, X_train, y_train):
        params = self.get_search_space(trial)
        model = self.create_model(params)
        
        f1_scorer = make_scorer(f1_score, average="weighted")
        scores = cross_val_score(
            model, X_train, y_train,
            cv=CV_FOLDS,
            scoring=f1_scorer,
            n_jobs=-1
        )
        
        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_f1_mean", scores.mean())
            
        return scores.mean()
        
    def tune_hyperparameters(self, X_train, y_train, n_trials=None):
        """Run Optuna tuning"""
        n_trials = n_trials or HYPERPARAMETER_SEARCH_TRIALS
        logger.info(f"Running {n_trials} trials for {self.model_type}")
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(f"{MLFLOW_EXPERIMENT_NAME}_tuning")
        
        self.study = optuna.create_study(
            direction="maximize",
            study_name=f"{self.model_type}_study"
        )
        
        with mlflow.start_run(run_name=f"{self.model_type}_tuning"):
            mlflow.log_param("model_type", self.model_type)
            
            self.study.optimize(
                lambda t: self.objective(t, X_train, y_train),
                n_trials=n_trials
            )
            
            self.best_params = self.study.best_params
            self.best_score = self.study.best_value
            
            mlflow.log_params(self.best_params)
            mlflow.log_metric("best_f1", self.best_score)
            
        return self.best_params, self.best_score
        
    def train_best_model(self, X_train, y_train, X_val, y_val):
        """Train model using best discovered parameters"""
        if not self.best_params:
            raise ValueError("Run tune_hyperparameters() first")
            
        model = self.create_model(self.best_params)
        model.fit(X_train, y_train)
        
        val_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, average="weighted")
        val_acc = accuracy_score(y_val, val_pred)
        
        logger.info(f"Validation Accuracy: {val_acc:.4f}")
        logger.info(f"Validation F1: {val_f1:.4f}")
        
        return model

def load_best_model_from_results(results_file=RESULTS_DIR / "training_results.json"):
    """Load the best model name from results.json file"""
    try:
        # Try to load from the provided path
        with open(results_file, 'r') as f:
            results = json.load(f)
            best_model = results.get('best_model')
            best_score = results.get('best_f1_score')
            
            if not best_model:
                raise ValueError("No 'best_model' field found in results.json")
                
            logger.info(f"="*60)
            logger.info(f"Loaded best model from {results_file}")
            logger.info(f"Best Model: {best_model}")
            logger.info(f"Baseline F1 Score: {best_score:.4f}")
            logger.info(f"="*60)
            
            return best_model, best_score
            
    except FileNotFoundError:
        logger.error(f"Could not find results.json file")
        raise FileNotFoundError(f"results.json not found. Please ensure the file exists.")
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing results.json: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading results.json: {e}")
        raise

def tune_best_model(model_type=None, results_file=RESULTS_DIR / "training_results.json"):    
    # If no model_type provided, load from results.json
    if model_type is None:
        model_type, baseline_score = load_best_model_from_results(results_file)
    else:
        logger.info(f"Using manually specified model: {model_type}")
        baseline_score = None
    
    logger.info("="*60)
    logger.info(f"Hyperparameter Tuning → {model_type}")
    logger.info("="*60)
    
    # Load and prepare data
    data_splits, _ = prepare_data_for_training()
    X_train, y_train, _ = data_splits['train']
    X_val, y_val, _ = data_splits['val']
    
    # TF-IDF vectorizer (same as model_training)
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=2,
        max_df=0.95
    )
    
    X_train_tf = vectorizer.fit_transform(X_train)
    X_val_tf = vectorizer.transform(X_val)
    
    # Run hyperparameter tuning
    tuner = HyperparameterTuner(model_type)
    best_params, best_score = tuner.tune_hyperparameters(X_train_tf, y_train)
    
    # Train best model
    best_model = tuner.train_best_model(X_train_tf, y_train, X_val_tf, y_val)
    

    # Save the tuned model ONLY if it improves over baseline
    save_model = True

    if baseline_score is not None:
        improvement = best_score - baseline_score
        if improvement <= 0:
            save_model = False
            logger.info(f"NO IMPROVEMENT → Tuned model NOT saved")
            logger.info(f"Baseline F1: {baseline_score:.4f}")
            logger.info(f"Tuned F1: {best_score:.4f}")

    if save_model:
        model_path = MODEL_DIR / f"{model_type}_tuned.pkl"
        joblib.dump(best_model, model_path)
        logger.info(f"Tuned model saved to {model_path}")
    else:
        logger.info(f"Tuned model not saved due to no improvement over baseline.")
        model_path = None

    # # Save the tuned model
    # model_path = MODEL_DIR / f"{model_type}_tuned.pkl"
    # joblib.dump(best_model, model_path)
    # logger.info(f"Tuned model saved to {model_path}")
    
    # Save tuning results
    tuning_results = {
        "model_type": model_type,
        "baseline_f1_score": baseline_score,
        "tuned_f1_score": best_score,
        "improvement": best_score - baseline_score if baseline_score else None,
        "best_params": best_params,
        "model_path": str(model_path)
    }
    
    tuning_results_path = MODEL_DIR / f"{model_type}_tuning_results.json"
    with open(tuning_results_path, 'w') as f:
        json.dump(tuning_results, f, indent=2, default=str)
    logger.info(f"Tuning results saved to {tuning_results_path}")
    
    if baseline_score:
        improvement = best_score - baseline_score
        logger.info("="*60)
        logger.info(f"TUNING SUMMARY")
        logger.info(f"Baseline F1: {baseline_score:.4f}")
        logger.info(f"Tuned F1: {best_score:.4f}")
        logger.info(f"Improvement: {improvement:+.4f} ({improvement/baseline_score*100:+.1f}%)")
        logger.info("="*60)
    
    return best_model, best_params

if __name__ == "__main__":
    import sys
    
    # Check if a model type was provided via command line
    if len(sys.argv) > 1:
        # If argument is a path to results.json
        if sys.argv[1].endswith('.json'):
            tune_best_model(results_file=sys.argv[1])
        else:
            # If argument is a model type (for manual override)
            tune_best_model(model_type=sys.argv[1])
    else:
        # Default: auto-select from results.json
        tune_best_model()