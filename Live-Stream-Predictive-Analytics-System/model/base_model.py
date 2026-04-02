"""
model/base_model.py
====================
Abstract base class that every prediction model must implement.
This ensures the pipeline can swap XGBoost for LSTM (or any future model)
without changing any other code.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseTimeSeriesModel(ABC):
    """
    Minimal interface for single-step-ahead time-series predictors.

    Concrete subclasses
    -------------------
    - XGBoostPredictor  (model/xgboost_model.py)
    - LSTMPredictor     (model/lstm_model.py)
    """

    @abstractmethod
    def fit(self, history: np.ndarray) -> None:
        """
        Train (or retrain) the model on the full observed history.

        Parameters
        ----------
        history : np.ndarray, shape (n,)
            All values seen so far, in chronological order.
        """

    @abstractmethod
    def predict(self, history: np.ndarray) -> float:
        """
        Return a one-step-ahead prediction.

        Parameters
        ----------
        history : np.ndarray, shape (n,)
            All values seen so far, in chronological order.
            ``history[-1]`` is the most recent observation.

        Returns
        -------
        float
            The predicted value for the next time step.
        """

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """True once ``fit`` has been called successfully at least once."""
