# Code to run model inference with trained models
import os

import joblib
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error


def calculate_rmse(y_true, y_pred):
    """Calculate the Root Mean Squared Error (RMSE) between the true and predicted values."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def confidence_interval_t_score(y_true, y_pred, confidence=0.95):
    """
    Calculate the confidence interval for RMSE using the t-distribution.

    Args:
    - y_true: Array of true values.
    - y_pred: Array of predicted values.
    - confidence: Confidence level for the interval.

    Returns:
    - A tuple representing the lower and upper bounds of the confidence interval.
    """
    squared_errors = (y_pred - y_true) ** 2
    interval = stats.t.interval(
        confidence,
        len(squared_errors) - 1,
        loc=squared_errors.mean(),
        scale=stats.sem(squared_errors),
    )
    return np.sqrt(interval)


def confidence_interval_bootstrap(y_true, y_pred, n_bootstraps=1000, confidence=0.95):
    """
    Calculate the confidence interval for RMSE using the bootstrap method.

    Args:
    - y_true: Array of true values.
    - y_pred: Array of predicted values.
    - n_bootstraps: Number of bootstrap samples to generate.
    - confidence: Confidence level for the interval.

    Returns:
    - A tuple representing the lower and upper bounds of the confidence interval.
    """
    bootstrap_rmses = []

    for _ in range(n_bootstraps):
        bootstrap_indices = np.random.choice(
            range(len(y_true)), size=len(y_true), replace=True
        )
        bootstrap_predictions = y_pred[bootstrap_indices]
        bootstrap_actuals = y_true.iloc[bootstrap_indices]
        bootstrap_rmse = np.sqrt(
            np.mean((bootstrap_predictions - bootstrap_actuals) ** 2)
        )
        bootstrap_rmses.append(bootstrap_rmse)

    lower_bound = np.percentile(bootstrap_rmses, (1 - confidence) / 2 * 100)
    upper_bound = np.percentile(bootstrap_rmses, (1 + confidence) / 2 * 100)

    return lower_bound, upper_bound


def save_model(model, file_path):
    """
    Save the trained model to a file.

    Args:
    - model: Trained model to be saved.
    - file_path: Path where the model should be saved (including filename and extension).
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")


def load_model(file_path):
    """
    Load a trained model from a file.

    Args:
    - file_path: Path to the saved model file.

    Returns:
    - Loaded model.
    """
    if os.path.exists(file_path):
        model = joblib.load(file_path)
        print(f"Model loaded from {file_path}")
        return model
    else:
        raise FileNotFoundError(f"No model found at {file_path}")
