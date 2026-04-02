"""
model/lstm_model.py
====================
LSTM-backed single-step-ahead predictor (requires TensorFlow ≥ 2.x).

Architecture
------------
  Input  →  LSTM(64, return_sequences=True)  →  Dropout(0.2)
         →  LSTM(32)                          →  Dropout(0.2)
         →  Dense(16, relu)                   →  Dense(1)

All values are z-score normalised before training / inference so the model
is insensitive to the absolute scale of the signal.

Dependencies
------------
  pip install tensorflow        (or tensorflow-cpu for CPU-only environments)
"""

import logging
from typing import Optional

import numpy as np

from model.base_model import BaseTimeSeriesModel

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _z_normalize(arr: np.ndarray):
    """
    Z-score normalise an array.

    Returns
    -------
    normalised : np.ndarray
    mu         : float
    sigma      : float   (never zero — floor at 1e-8)
    """
    mu    = float(np.mean(arr))
    sigma = float(np.std(arr)) + 1e-8
    return (arr - mu) / sigma, mu, sigma


def _make_sequences(data: np.ndarray, window: int):
    """
    Convert a 1-D time series into overlapping (X, y) sequence pairs.

    Returns
    -------
    X : np.ndarray, shape (n - window, window, 1)
    y : np.ndarray, shape (n - window,)
    """
    xs, ys = [], []
    for i in range(window, len(data)):
        xs.append(data[i - window : i])
        ys.append(data[i])
    X = np.array(xs, dtype=np.float32)[..., np.newaxis]   # (N, W, 1)
    y = np.array(ys, dtype=np.float32)                     # (N,)
    return X, y


# ------------------------------------------------------------------ #
# Model                                                                #
# ------------------------------------------------------------------ #

class LSTMPredictor(BaseTimeSeriesModel):
    """
    Single-step-ahead predictor backed by a two-layer stacked LSTM.

    Parameters
    ----------
    config : ModelConfig
        Must expose ``window_size`` (int).
    """

    def __init__(self, config) -> None:
        self.window   = config.window_size
        self._model   = None
        self._trained = False
        # normalisation statistics updated each time fit() is called
        self._mu:    float = 0.0
        self._sigma: float = 1.0

    # ------------------------------------------------------------------ #

    @property
    def is_trained(self) -> bool:
        return self._trained

    # ------------------------------------------------------------------ #

    def _build_keras_model(self):
        """Construct and compile the Keras model (called once)."""
        try:
            import tensorflow as tf  # type: ignore  # noqa: F401
            from tensorflow import keras  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow is not installed.  Run:  pip install tensorflow"
            ) from exc

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(self.window, 1)),
                keras.layers.LSTM(64, return_sequences=True),
                keras.layers.Dropout(0.2),
                keras.layers.LSTM(32),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dense(1),
            ],
            name="lstm_predictor",
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="mse",
        )
        return model

    # ------------------------------------------------------------------ #

    def fit(self, history: np.ndarray) -> None:
        """
        (Re-)train the LSTM on the full observed history.

        Requires at least ``window + 10`` samples before doing anything.
        """
        if len(history) < self.window + 10:
            logger.debug(
                "Skipping LSTM training: only %d samples", len(history)
            )
            return

        if self._model is None:
            self._model = self._build_keras_model()

        norm, self._mu, self._sigma = _z_normalize(history)
        X, y = _make_sequences(norm, self.window)

        self._model.fit(
            X, y,
            epochs=5,
            batch_size=min(32, len(y)),
            verbose=0,
        )
        self._trained = True
        logger.info(
            "LSTM retrained  sequences=%d  window=%d", len(y), self.window
        )

    def predict(self, history: np.ndarray) -> float:
        """
        Return a one-step-ahead forecast.

        Falls back to the last observed value before the model is trained.
        """
        if not self._trained or len(history) < self.window:
            return float(history[-1])

        seq = (history[-self.window :] - self._mu) / self._sigma
        x   = seq.reshape(1, self.window, 1).astype(np.float32)
        norm_pred = float(self._model.predict(x, verbose=0)[0][0])
        return float(norm_pred * self._sigma + self._mu)
