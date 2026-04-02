"""
data_generator/producer.py
===========================
Wraps TimeSeriesGenerator and publishes each DataPoint to a Kafka topic.

This module is only required when running with --mode kafka.
kafka-python must be installed:  pip install kafka-python
"""

import logging
import threading
from typing import Optional

from data_generator.generator import TimeSeriesGenerator

logger = logging.getLogger(__name__)


class KafkaDataProducer:
    """
    Runs the data generator in a background thread and sends every
    DataPoint (JSON-serialised) to the configured Kafka topic.

    Parameters
    ----------
    generator  : TimeSeriesGenerator — the data source.
    kafka_config : KafkaConfig       — broker / topic settings.
    """

    def __init__(self, generator: TimeSeriesGenerator, kafka_config) -> None:
        self.generator = generator
        self.kafka_config = kafka_config
        self._producer = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_producer(self):
        """Lazily build the KafkaProducer on first use."""
        if self._producer is not None:
            return self._producer
        try:
            from kafka import KafkaProducer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "kafka-python is not installed.  Run:  pip install kafka-python"
            ) from exc

        self._producer = KafkaProducer(
            bootstrap_servers=self.kafka_config.bootstrap_servers,
            value_serializer=lambda v: v.encode("utf-8"),
            acks="all",
            retries=3,
        )
        logger.info(
            "KafkaProducer connected to %s", self.kafka_config.bootstrap_servers
        )
        return self._producer

    def _publish_loop(self) -> None:
        producer = self._build_producer()
        for point in self.generator.stream(stop_event=self._stop_event):
            try:
                producer.send(self.kafka_config.topic, value=point.to_json())
                logger.debug(
                    "Published  seq_id=%-6d  value=%.4f",
                    point.sequence_id,
                    point.value,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to publish message: %s", exc)
        producer.flush()
        producer.close()
        logger.info("KafkaProducer closed")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the background publisher thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._publish_loop,
            daemon=True,
            name="KafkaProducerThread",
        )
        self._thread.start()
        logger.info(
            "Kafka producer started  topic='%s'", self.kafka_config.topic
        )

    def stop(self) -> None:
        """Signal the publisher thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        logger.info("Kafka producer stopped")
