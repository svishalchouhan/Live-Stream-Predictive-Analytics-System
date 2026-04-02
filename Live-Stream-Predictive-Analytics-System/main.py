"""
main.py — Entry point for the Live-Stream Predictive Analytics System
======================================================================

Quick-start examples
--------------------

    # Default: simple pipeline, XGBoost, stock signal at 2 Hz
    python main.py

    # LSTM model on sensor data at 5 Hz
    python main.py --model lstm --signal sensor --rate 5

    # Composite sine signal (easy to verify model accuracy)
    python main.py --signal sine --noise 0.02

    # Kafka-backed pipeline (requires running broker — see docker-compose.yml)
    python main.py --mode kafka --signal stock

    # Verbose logging to see every training event
    python main.py --log-level DEBUG
"""

import argparse
import logging
import signal
import sys

from config import (
    AppConfig,
    DataGeneratorConfig,
    KafkaConfig,
    ModelConfig,
    PipelineConfig,
)
from streaming.pipeline import KafkaPipeline, SimplePipeline
from dashboard.visualizer import LiveDashboard


# ================================================================== #
#  CLI                                                                 #
# ================================================================== #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Live-Stream Predictive Analytics System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Pipeline
    p.add_argument(
        "--mode", choices=["simple", "kafka"], default="simple",
        help="Pipeline transport: in-process queues or Apache Kafka",
    )

    # Data generator
    p.add_argument(
        "--signal", choices=["stock", "sensor", "sine"], default="stock",
        help="Synthetic signal type",
    )
    p.add_argument(
        "--rate", type=float, default=2.0, metavar="HZ",
        help="Data generation rate (samples per second)",
    )
    p.add_argument(
        "--noise", type=float, default=0.05,
        help="Relative noise amplitude",
    )

    # Model
    p.add_argument(
        "--model", choices=["xgboost", "lstm"], default="xgboost",
        help="Prediction model",
    )
    p.add_argument(
        "--window", type=int, default=20,
        help="Input window size (number of past values fed to the model)",
    )
    p.add_argument(
        "--min-samples", type=int, default=60,
        help="Minimum samples collected before the first training run",
    )
    p.add_argument(
        "--retrain-interval", type=int, default=50,
        help="Retrain every N new samples after the initial training",
    )

    # Dashboard
    p.add_argument(
        "--display-points", type=int, default=200,
        help="Rolling window of data points shown in the live dashboard",
    )

    # Kafka (only relevant with --mode kafka)
    p.add_argument(
        "--kafka-server", default="localhost:9092",
        help="Kafka bootstrap server address",
    )
    p.add_argument(
        "--kafka-topic", default="live_stream_data",
        help="Kafka topic name",
    )

    # Logging
    p.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO",
        help="Console log verbosity",
    )

    return p


def _args_to_config(args: argparse.Namespace) -> AppConfig:
    return AppConfig(
        data_generator=DataGeneratorConfig(
            signal_type=args.signal,
            generation_rate_hz=args.rate,
            noise_level=args.noise,
        ),
        model=ModelConfig(
            model_type=args.model,
            window_size=args.window,
            min_train_samples=args.min_samples,
            retrain_interval=args.retrain_interval,
        ),
        kafka=KafkaConfig(
            bootstrap_servers=args.kafka_server,
            topic=args.kafka_topic,
        ),
        pipeline=PipelineConfig(
            mode=args.mode,
            max_display_points=args.display_points,
        ),
        log_level=args.log_level,
    )


# ================================================================== #
#  Main                                                                #
# ================================================================== #

def main() -> None:
    args   = _build_parser().parse_args()
    config = _args_to_config(args)

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logger = logging.getLogger("main")

    # ---- Build pipeline ---------------------------------------------- #
    if config.pipeline.mode == "kafka":
        pipeline = KafkaPipeline(config)
    else:
        pipeline = SimplePipeline(config)

    pipeline.start()

    logger.info(
        "Pipeline started  [mode=%-6s  signal=%-6s  model=%-8s  rate=%.1f Hz]",
        config.pipeline.mode,
        config.data_generator.signal_type,
        config.model.model_type,
        config.data_generator.generation_rate_hz,
    )
    logger.info(
        "Collecting %d bootstrap samples before first training …",
        config.model.min_train_samples,
    )

    # ---- Graceful shutdown on SIGINT / SIGTERM ----------------------- #
    _shutdown_requested = {"flag": False}

    def _shutdown(sig, frame):               # noqa: ANN001
        if not _shutdown_requested["flag"]:
            _shutdown_requested["flag"] = True
            logger.info("Shutdown requested — stopping pipeline …")
            pipeline.stop()
            sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ---- Launch web dashboard (blocks until Ctrl-C) ------------------- #
    dashboard = LiveDashboard(
        result_queue=pipeline.result_queue,
        max_points=config.pipeline.max_display_points,
        host="0.0.0.0",
        port=8050,
    )
    dashboard.run()

    # Reached when the user closes the dashboard window
    pipeline.stop()
    logger.info("System shut down cleanly")


if __name__ == "__main__":
    main()
