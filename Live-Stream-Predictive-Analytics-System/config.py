"""
config.py — Central configuration dataclasses for the Live-Stream Predictive Analytics System.
All runtime knobs are controlled from here (or via CLI flags in main.py).
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DataGeneratorConfig:
    """Settings for the synthetic time-series data generator."""

    # Signal type to simulate
    signal_type: Literal["stock", "sensor", "sine"] = "stock"

    # How fast data is produced (samples per second)
    generation_rate_hz: float = 2.0

    # Relative noise amplitude
    noise_level: float = 0.05

    # Drift / trend per time-step
    trend_factor: float = 0.001

    # Starting value of the generated signal
    initial_value: float = 100.0

    # Random seed for reproducibility
    seed: int = 42


@dataclass
class ModelConfig:
    """Settings for the prediction model."""

    model_type: Literal["xgboost", "lstm"] = "xgboost"

    # Number of past values fed as input features
    window_size: int = 20

    # Steps ahead to forecast (currently only 1-step-ahead is implemented)
    prediction_horizon: int = 1

    # Retrain the model every N new data points (after initial training)
    retrain_interval: int = 50

    # Minimum samples required before the first training run
    min_train_samples: int = 60


@dataclass
class KafkaConfig:
    """Kafka connection settings (only used in --mode kafka)."""

    bootstrap_servers: str = "localhost:9092"
    topic: str = "live_stream_data"
    group_id: str = "predictor_group"
    auto_offset_reset: str = "latest"


@dataclass
class PipelineConfig:
    """Settings that control how results are routed through the pipeline."""

    # "simple" = in-process queues; "kafka" = Apache Kafka
    mode: Literal["simple", "kafka"] = "simple"

    # Maximum items held in the internal queues before back-pressure kicks in
    queue_maxsize: int = 500

    # Maximum data points kept on the live dashboard
    max_display_points: int = 200


@dataclass
class AppConfig:
    """Top-level application configuration — aggregates all sub-configs."""

    data_generator: DataGeneratorConfig = field(default_factory=DataGeneratorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    log_level: str = "INFO"
