"""
data_generator/generator.py
============================
Synthetic time-series data generator.

Three signal types are supported:
  • stock   — Geometric Brownian Motion (realistic price simulation)
  • sensor  — Trend + multi-period seasonality + noise + rare spikes
  • sine    — Composite sine wave with injected noise

Usage
-----
    from data_generator.generator import TimeSeriesGenerator
    from config import DataGeneratorConfig
    import threading

    stop = threading.Event()
    gen  = TimeSeriesGenerator(DataGeneratorConfig(signal_type="stock"))
    for point in gen.stream(stop_event=stop):
        print(point)
"""

import json
import threading
import time
from dataclasses import asdict, dataclass
from typing import Generator, Optional

import numpy as np


@dataclass
class DataPoint:
    """A single timestamped observation produced by the generator."""

    timestamp: float    # Unix epoch (seconds)
    value: float        # The signal value
    signal_type: str    # Which signal produced it
    sequence_id: int    # Monotonically increasing index

    # ------------------------------------------------------------------ #
    # Serialisation helpers                                                #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "DataPoint":
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> "DataPoint":
        return cls.from_dict(json.loads(s))


class TimeSeriesGenerator:
    """
    Generates an infinite stream of synthetic DataPoints.

    Parameters
    ----------
    config : DataGeneratorConfig
        Central configuration object (see config.py).
    """

    def __init__(self, config) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.current_value: float = float(config.initial_value)
        self.sequence_id: int = 0
        self._t: int = 0  # internal discrete time counter

    # ------------------------------------------------------------------ #
    # Signal generators                                                    #
    # ------------------------------------------------------------------ #

    def _geometric_brownian_motion(self) -> float:
        """
        Geometric Brownian Motion — models realistic stock-price dynamics.

            S(t+dt) = S(t) * exp((μ - ½σ²)dt + σ√dt · Z)
        """
        dt = 1.0 / max(self.config.generation_rate_hz, 1e-6)
        mu = self.config.trend_factor
        sigma = self.config.noise_level
        z = float(self.rng.normal(0.0, 1.0))
        self.current_value *= float(
            np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
        )
        return self.current_value

    def _sensor_reading(self) -> float:
        """
        Industrial sensor reading.
        Combines a linear trend, two-frequency seasonality, Gaussian noise,
        and rare spike anomalies (~1 % of samples).
        """
        t = self._t
        trend = self.config.trend_factor * t
        seasonal = (
            10.0 * np.sin(2.0 * np.pi * t / 100.0)
            + 5.0 * np.sin(2.0 * np.pi * t / 20.0)
        )
        noise = float(
            self.rng.normal(0.0, self.config.noise_level * self.config.initial_value)
        )
        spike = (
            float(self.rng.uniform(-25.0, 25.0))
            if self.rng.random() < 0.01
            else 0.0
        )
        return float(self.config.initial_value + trend + seasonal + noise + spike)

    def _sine_wave(self) -> float:
        """
        Composite sine wave — good for verifying model accuracy on
        deterministic, predictable patterns.
        """
        t = self._t
        sig = (
            10.0 * np.sin(2.0 * np.pi * t / 50.0)
            + 5.0 * np.sin(2.0 * np.pi * t / 20.0)
        )
        noise = float(self.rng.normal(0.0, self.config.noise_level * 5.0))
        return float(self.config.initial_value + sig + noise)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def next(self) -> DataPoint:
        """Generate and return the next DataPoint."""
        _dispatch = {
            "stock":  self._geometric_brownian_motion,
            "sensor": self._sensor_reading,
            "sine":   self._sine_wave,
        }
        value_fn = _dispatch.get(self.config.signal_type, self._geometric_brownian_motion)
        value = value_fn()

        point = DataPoint(
            timestamp=time.time(),
            value=value,
            signal_type=self.config.signal_type,
            sequence_id=self.sequence_id,
        )
        self.sequence_id += 1
        self._t += 1
        return point

    def stream(
        self,
        stop_event: Optional[threading.Event] = None,
    ) -> Generator[DataPoint, None, None]:
        """
        Yield DataPoints continuously at ``config.generation_rate_hz``.

        Parameters
        ----------
        stop_event : threading.Event, optional
            When set, the generator stops after the current sleep interval.
        """
        interval = 1.0 / max(self.config.generation_rate_hz, 1e-6)
        while stop_event is None or not stop_event.is_set():
            yield self.next()
            time.sleep(interval)
