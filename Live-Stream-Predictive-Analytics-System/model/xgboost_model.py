"""
model/xgboost_model.py
=======================
XGBoost-backed single-step-ahead predictor.

Feature engineering
-------------------
For each training / inference window of length W the following features
are computed:

  • W lag values              (lag_1 … lag_W   — i.e. the raw window)
  • Rolling mean              (over the full window)
  • Rolling std               (over the full window)
  • Rolling min / max
  • Short-term mean           (last W//2 values)
  • Momentum                  (last − first value in the window)

Total features = W + 7

Dependencies
------------
  pip install xgboost            (preferred)
  — falls back to scikit-learn GradientBoostingRegressor if xgboost is absent.
"""

import logging

import numpy as np

from model.base_model import BaseTimeSeriesModel

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Feature helpers                                                      #
# ------------------------------------------------------------------ #

def _extract_features(window: np.ndarray) -> np.ndarray:
    """
    Build a 1-D feature vector from a fixed-length window of values.

    Parameters
    ----------
    window : np.ndarray, shape (W,)

    Returns
    -------
    np.ndarray, shape (W + 7,)
    """
    w = window.astype(np.float32)
    half = max(len(w) // 2, 1)

    lags      = w.tolist()
    roll_mean = float(np.mean(w))
    roll_std  = float(np.std(w)) + 1e-8
    roll_min  = float(np.min(w))
    roll_max  = float(np.max(w))
    short_mean = float(np.mean(w[-half:]))
    long_mean  = float(np.mean(w))
    momentum   = float(w[-1] - w[0])

    return np.array(
        lags + [roll_mean, roll_std, roll_min, roll_max, short_mean, long_mean, momentum],
        dtype=np.float32,
    )


def _build_supervised_dataset(history: np.ndarray, window: int):
    """
    Slide a window over ``history`` to produce (X, y) pairs for supervised
    regression.

    Parameters
    ----------
    history : np.ndarray, shape (n,)
    window  : int

    Returns
    -------
    X : np.ndarray, shape (n - window, window + 7)
    y : np.ndarray, shape (n - window,)
    """
    rows_x, rows_y = [], []
    for i in range(window, len(history)):
        rows_x.append(_extract_features(history[i - window : i]))
        rows_y.append(float(history[i]))
    return np.array(rows_x, dtype=np.float32), np.array(rows_y, dtype=np.float32)


# ------------------------------------------------------------------ #
# Model                                                                #
# ------------------------------------------------------------------ #

class XGBoostPredictor(BaseTimeSeriesModel):
    """
    Single-step-ahead predictor backed by XGBoost (or sklearn fallback).

    Parameters
    ----------
    config : ModelConfig
        Must expose ``window_size`` (int).
    """

    def __init__(self, config) -> None:
        self.window = config.window_size
        self._model = None
        self._trained = False

    # ------------------------------------------------------------------ #

    @property
    def is_trained(self) -> bool:
        return self._trained

    # ------------------------------------------------------------------ #

    def _build_model(self):
        """Instantiate XGBRegressor, or fall back to sklearn."""
        try:
            import xgboost as xgb  # type: ignore

            return xgb.XGBRegressor(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                n_jobs=-1,
                verbosity=0,
            )
        except ImportError:
            logger.warning(
                "xgboost not installed — falling back to "
                "sklearn GradientBoostingRegressor"
            )
            from sklearn.ensemble import GradientBoostingRegressor  # type: ignore

            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
            )

    # ------------------------------------------------------------------ #

    def fit(self, history: np.ndarray) -> None:
        """
        (Re-)train the model on the full observed history.

        Does nothing if there are fewer than ``window + 1`` samples.
        """
        if len(history) <= self.window:
            logger.debug(
                "Skipping training: only %d samples (need > %d)",
                len(history),
                self.window,
            )
            return

        X, y = _build_supervised_dataset(history, self.window)

        if self._model is None:
            self._model = self._build_model()

        self._model.fit(X, y)
        self._trained = True
        logger.info(
            "XGBoost retrained  samples=%d  features=%d", len(y), X.shape[1]
        )

    def predict(self, history: np.ndarray) -> float:
        """
        Return a one-step-ahead forecast.

        Falls back to the last observed value if the model has not been
        trained yet or if there is insufficient history.
        """
        if not self._trained or len(history) < self.window:
            return float(history[-1])  # naïve last-value fallback

        features = _extract_features(history[-self.window :]).reshape(1, -1)
        return float(self._model.predict(features)[0])
