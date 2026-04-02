"""
streaming/results.py
=====================
Shared dataclass for prediction results.

Kept in a separate module so both ``streaming/pipeline.py`` and
``dashboard/visualizer.py`` can import it without circular dependencies.
"""

from dataclasses import dataclass


@dataclass
class PredictionResult:
    """Carries everything needed for logging, downstream routing, and display."""

    timestamp:        float   # Unix epoch when the original DataPoint was generated
    sequence_id:      int     # Matches DataPoint.sequence_id
    actual_value:     float   # The ground-truth observation
    predicted_value:  float   # Model's one-step-ahead forecast
    prediction_error: float   # abs(actual - predicted)
    model_type:       str     # "xgboost" | "lstm"
