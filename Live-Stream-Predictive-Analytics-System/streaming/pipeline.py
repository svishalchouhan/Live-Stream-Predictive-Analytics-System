"""
streaming/pipeline.py
======================
Orchestrates the end-to-end data flow from generator → model → results.

Two pipeline implementations are provided:

  SimplePipeline
  --------------
  Everything runs in-process using Python thread-safe queues.
  No external infrastructure required.

    Generator Thread  →  data_queue  →  PredictorWorker Thread  →  result_queue

  KafkaPipeline
  -------------
  The generator pushes to Kafka; a consumer pulls from Kafka.
  Requires a running Kafka broker (see docker-compose.yml).

    Generator Thread  →  KafkaProducer  →  Kafka  →  KafkaConsumer  →  data_queue
                      →  PredictorWorker Thread  →  result_queue
"""

import logging
import queue
import threading
from typing import List, Optional

import numpy as np

from data_generator.generator import DataPoint, TimeSeriesGenerator
from model.xgboost_model import XGBoostPredictor
from model.lstm_model import LSTMPredictor
from streaming.results import PredictionResult

logger = logging.getLogger(__name__)


# ================================================================== #
#  Predictor Worker                                                    #
# ================================================================== #

class PredictorWorker:
    """
    Reads DataPoints from ``in_queue``, maintains a running history,
    triggers periodic model retraining, emits PredictionResults to
    ``out_queue``.

    Parameters
    ----------
    model_config : ModelConfig
    in_queue     : queue.Queue[DataPoint]
    out_queue    : queue.Queue[PredictionResult]
    """

    def __init__(
        self,
        model_config,
        in_queue:  queue.Queue,
        out_queue: queue.Queue,
    ) -> None:
        self.config   = model_config
        self.in_queue  = in_queue
        self.out_queue = out_queue

        self._history: List[float] = []
        self._model = self._create_model()
        self._samples_since_retrain: int = 0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #

    def _create_model(self):
        if self.config.model_type == "lstm":
            return LSTMPredictor(self.config)
        return XGBoostPredictor(self.config)

    # ------------------------------------------------------------------ #

    def _should_retrain(self) -> bool:
        n = len(self._history)
        if n == self.config.min_train_samples:
            return True                          # first training run
        return (
            n > self.config.min_train_samples
            and self._samples_since_retrain >= self.config.retrain_interval
        )

    # ------------------------------------------------------------------ #

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                point: DataPoint = self.in_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            self._history.append(point.value)
            history_arr = np.array(self._history, dtype=np.float64)

            # ---- periodic retraining ---------------------------------- #
            if self._should_retrain():
                try:
                    self._model.fit(history_arr)
                    self._samples_since_retrain = 0
                    logger.info(
                        "Model retrained  samples=%-6d  seq_id=%d",
                        len(self._history),
                        point.sequence_id,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("Retraining failed: %s", exc)

            self._samples_since_retrain += 1

            # ---- prediction ------------------------------------------ #
            predicted = self._model.predict(history_arr)
            error     = abs(point.value - predicted)

            result = PredictionResult(
                timestamp        = point.timestamp,
                sequence_id      = point.sequence_id,
                actual_value     = point.value,
                predicted_value  = predicted,
                prediction_error = error,
                model_type       = self.config.model_type,
            )

            # Non-blocking put: drop oldest result if queue is full
            try:
                self.out_queue.put_nowait(result)
            except queue.Full:
                try:
                    self.out_queue.get_nowait()
                    self.out_queue.put_nowait(result)
                except queue.Empty:
                    pass

    # ------------------------------------------------------------------ #

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="PredictorWorkerThread",
        )
        self._thread.start()
        logger.info(
            "PredictorWorker started  model=%s  window=%d",
            self.config.model_type,
            self.config.window_size,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        logger.info("PredictorWorker stopped")


# ================================================================== #
#  Simple (in-process) Pipeline                                        #
# ================================================================== #

class SimplePipeline:
    """
    Fully in-process pipeline — no Kafka required.

    Thread layout
    -------------
    GeneratorProducerThread  →  data_queue  →  PredictorWorkerThread
                                                      ↓
                                              result_queue  ← read by dashboard
    """

    def __init__(self, app_config) -> None:
        self.config = app_config

        self.data_queue:   queue.Queue = queue.Queue(
            maxsize=app_config.pipeline.queue_maxsize
        )
        self.result_queue: queue.Queue = queue.Queue(
            maxsize=app_config.pipeline.queue_maxsize
        )

        self._stop_event = threading.Event()
        self._generator  = TimeSeriesGenerator(app_config.data_generator)
        self._predictor  = PredictorWorker(
            app_config.model, self.data_queue, self.result_queue
        )
        self._producer_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #

    def _producer_loop(self) -> None:
        for point in self._generator.stream(stop_event=self._stop_event):
            try:
                self.data_queue.put(point, block=True, timeout=1.0)
                logger.debug(
                    "Enqueued  seq_id=%-6d  value=%.4f",
                    point.sequence_id,
                    point.value,
                )
            except queue.Full:
                logger.warning(
                    "Data queue full — dropping seq_id=%d", point.sequence_id
                )

    def start(self) -> None:
        self._stop_event.clear()
        self._producer_thread = threading.Thread(
            target=self._producer_loop,
            daemon=True,
            name="GeneratorProducerThread",
        )
        self._producer_thread.start()
        self._predictor.start()
        logger.info("SimplePipeline started")

    def stop(self) -> None:
        self._stop_event.set()
        self._predictor.stop()
        if self._producer_thread is not None:
            self._producer_thread.join(timeout=5)
        logger.info("SimplePipeline stopped")


# ================================================================== #
#  Kafka Pipeline                                                      #
# ================================================================== #

class KafkaPipeline:
    """
    Kafka-backed pipeline.

    Thread layout
    -------------
    GeneratorThread → KafkaProducer → [Kafka broker] → KafkaConsumerThread
                                                              ↓
                                                        data_queue
                                                              ↓
                                                    PredictorWorkerThread
                                                              ↓
                                                        result_queue  ← dashboard
    """

    def __init__(self, app_config) -> None:
        self.config = app_config

        self.data_queue:   queue.Queue = queue.Queue(
            maxsize=app_config.pipeline.queue_maxsize
        )
        self.result_queue: queue.Queue = queue.Queue(
            maxsize=app_config.pipeline.queue_maxsize
        )

        # Import Kafka-dependent classes only when this pipeline is used
        from data_generator.producer import KafkaDataProducer
        from streaming.consumer import KafkaDataConsumer

        generator = TimeSeriesGenerator(app_config.data_generator)
        self._kafka_producer = KafkaDataProducer(generator, app_config.kafka)
        self._kafka_consumer = KafkaDataConsumer(app_config.kafka, self.data_queue)
        self._predictor      = PredictorWorker(
            app_config.model, self.data_queue, self.result_queue
        )

    # ------------------------------------------------------------------ #

    def start(self) -> None:
        self._kafka_producer.start()
        self._kafka_consumer.start()
        self._predictor.start()
        logger.info("KafkaPipeline started")

    def stop(self) -> None:
        self._kafka_producer.stop()
        self._kafka_consumer.stop()
        self._predictor.stop()
        logger.info("KafkaPipeline stopped")
