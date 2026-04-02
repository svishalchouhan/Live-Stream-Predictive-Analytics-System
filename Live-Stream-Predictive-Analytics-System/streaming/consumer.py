"""
streaming/consumer.py
======================
Reads DataPoints from a Kafka topic and places them in a thread-safe queue
for the predictor worker to consume.

This module is only required when running with --mode kafka.
kafka-python must be installed:  pip install kafka-python
"""

import logging
import queue
import threading
from typing import Optional

from data_generator.generator import DataPoint

logger = logging.getLogger(__name__)


class KafkaDataConsumer:
    """
    Background thread that polls a Kafka topic and enqueues deserialised
    DataPoints for the predictor worker.

    Parameters
    ----------
    kafka_config : KafkaConfig
        Broker / topic settings.
    out_queue : queue.Queue
        Destination queue for incoming DataPoints.
    """

    def __init__(self, kafka_config, out_queue: queue.Queue) -> None:
        self.kafka_config = kafka_config
        self.out_queue    = out_queue
        self._consumer    = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event  = threading.Event()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_consumer(self):
        """Lazily create the KafkaConsumer on first use."""
        if self._consumer is not None:
            return self._consumer
        try:
            from kafka import KafkaConsumer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "kafka-python is not installed.  Run:  pip install kafka-python"
            ) from exc

        self._consumer = KafkaConsumer(
            self.kafka_config.topic,
            bootstrap_servers=self.kafka_config.bootstrap_servers,
            group_id=self.kafka_config.group_id,
            auto_offset_reset=self.kafka_config.auto_offset_reset,
            value_deserializer=lambda m: m.decode("utf-8"),
            # poll() returns after this many ms even if no messages arrived
            consumer_timeout_ms=500,
        )
        logger.info(
            "KafkaConsumer connected  topic='%s'", self.kafka_config.topic
        )
        return self._consumer

    def _consume_loop(self) -> None:
        consumer = self._build_consumer()
        while not self._stop_event.is_set():
            for message in consumer:
                if self._stop_event.is_set():
                    break
                try:
                    point = DataPoint.from_json(message.value)
                    self.out_queue.put(point, block=True, timeout=1.0)
                    logger.debug("Consumed  seq_id=%d", point.sequence_id)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed to deserialise message: %s", exc)
        consumer.close()
        logger.info("KafkaConsumer closed")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the background consumer thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._consume_loop,
            daemon=True,
            name="KafkaConsumerThread",
        )
        self._thread.start()
        logger.info(
            "Kafka consumer started  topic='%s'", self.kafka_config.topic
        )

    def stop(self) -> None:
        """Signal the consumer thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        logger.info("Kafka consumer stopped")
