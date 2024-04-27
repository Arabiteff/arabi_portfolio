"""
Metrics module
--------------
"""
import numpy as np
import pandas as pd

__all__ = ["exp_forecast_accuracy", "forecast_accuracy", "bias", "identity"]


def identity(x_var):
    """Convert a list/array to an identic array

    Args:
        x_var (array or list): array or list to convert as np.array

    Returns:
        np.array
    """
    return np.asarray(x_var)


def forecast_accuracy(actual, predict, transform=identity) -> float:
    """
    Compute forecast accuracy between actual and prediction(transformed)
    ---


    Args:
    ----
        actual (np.array): Array of true/real values
        predict (np.array): Array of predicted values
        transform (function, optional): function to return an array. Defaults to identity.

    Returns:
    -------
        float
    """
    abs_diff = np.abs(transform(predict) - transform(actual))
    return 1.0 - np.sum(abs_diff) / np.sum(transform(actual))


def bias(actual, predict, transform=identity):
    """
    Compute bias between actuals/real values and forecasted values (transformed)
    ---

    Args:
    ----
        actual (np.array): Array of true/real values
        predict (np.array): Array of predicted values
        transform (function, optional): function to return an array. Defaults to identity.

    Returns:
    -------
        float
    """
    return np.sum(transform(predict) - transform(actual)) / np.sum(transform(actual))


def exp_forecast_accuracy(actual, predict):
    """Compute forecast accuracy between actuals/real values and forecasted values(in log)
    ---
    Args:
    ---
        actual (np.array): Array of true/real values
        predict (np.array): Array of log predicted values

    Returns:
    -------
        float
    """
    return forecast_accuracy(actual, predict, transform=np.exp)


