"""
dashboard/visualizer.py
========================
Plotly-Dash web dashboard for real-time display of predictions.

Layout
------
  • KPI bar     — Actual / Predicted / MAE / Samples / Model
  • Top chart   — Actual vs Predicted (line)
  • Bottom-left — Absolute error over time (line)
  • Bottom-right— Error distribution (histogram)

The page auto-refreshes every 750 ms via a dcc.Interval component.
A background thread continuously drains the result queue into thread-safe
deques so the Dash callbacks never block.

Open http://localhost:8050 in any browser after running main.py.
"""

import logging
import queue
import threading
from collections import deque
from typing import Deque

import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

from streaming.results import PredictionResult

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Colour tokens (dark theme)                                           #
# ------------------------------------------------------------------ #
_BG       = "#0f0f1a"
_CARD_BG  = "#16213e"
_BLUE     = "#2196F3"
_ORANGE   = "#FF5722"
_GREEN    = "#4CAF50"
_AMBER    = "#FF9800"
_TEXT     = "#e0e0e0"
_SUBTEXT  = "#9e9e9e"
_GRID     = "rgba(255,255,255,0.07)"


_AXIS_DEFAULTS = dict(gridcolor=_GRID, zerolinecolor=_GRID)


def _dark_layout(**kwargs) -> go.Layout:
    """Return a dark-themed go.Layout with sensible defaults."""
    # Merge axis defaults with any caller-supplied axis overrides
    for axis_key in ("xaxis", "yaxis"):
        if axis_key in kwargs:
            merged = dict(_AXIS_DEFAULTS)
            merged.update(kwargs[axis_key])
            kwargs[axis_key] = merged
        else:
            kwargs[axis_key] = dict(_AXIS_DEFAULTS)
    return go.Layout(
        paper_bgcolor=_CARD_BG,
        plot_bgcolor=_CARD_BG,
        font=dict(color=_TEXT, size=11),
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        **kwargs,
    )


# ================================================================== #
#  LiveDashboard                                                       #
# ================================================================== #

class LiveDashboard:
    """
    Plotly-Dash web dashboard.

    Parameters
    ----------
    result_queue : queue.Queue[PredictionResult]
        Source of live predictions from the pipeline.
    max_points : int
        Rolling window of data points kept for display.
    host : str
        Interface to bind to ("0.0.0.0" for all interfaces).
    port : int
        TCP port for the Dash dev server.
    """

    def __init__(
        self,
        result_queue: queue.Queue,
        max_points: int = 200,
        host: str = "127.0.0.1",
        port: int = 8050,
    ) -> None:
        self.result_queue = result_queue
        self.max_points   = max_points
        self.host         = host
        self.port         = port

        # Thread-safe rolling buffers
        self._lock        = threading.Lock()
        self._actuals:    Deque[float] = deque(maxlen=max_points)
        self._predictions:Deque[float] = deque(maxlen=max_points)
        self._errors:     Deque[float] = deque(maxlen=max_points)
        self._seq_ids:    Deque[int]   = deque(maxlen=max_points)
        self._model_type: str          = "—"

        self._stop_drain  = threading.Event()
        self._app         = self._build_app()

    # ------------------------------------------------------------------ #
    # Queue drain thread                                                   #
    # ------------------------------------------------------------------ #

    def _drain_loop(self) -> None:
        """Background thread: empties the result queue into the deques."""
        while not self._stop_drain.is_set():
            try:
                res: PredictionResult = self.result_queue.get(timeout=0.2)
                with self._lock:
                    self._actuals.append(res.actual_value)
                    self._predictions.append(res.predicted_value)
                    self._errors.append(res.prediction_error)
                    self._seq_ids.append(res.sequence_id)
                    self._model_type = res.model_type
            except queue.Empty:
                continue

    # ------------------------------------------------------------------ #
    # Dash app                                                             #
    # ------------------------------------------------------------------ #

    def _build_app(self) -> dash.Dash:
        app = dash.Dash(
            __name__,
            title="Live-Stream Predictive Analytics",
            update_title=None,
        )
        app.logger.setLevel(logging.WARNING)

        # ---- layout -------------------------------------------------- #
        app.layout = html.Div(
            style={"backgroundColor": _BG, "minHeight": "100vh",
                   "fontFamily": "'Segoe UI', sans-serif", "padding": "20px"},
            children=[
                # Header
                html.H2(
                    "Live-Stream Predictive Analytics",
                    style={"color": _TEXT, "marginBottom": "4px"},
                ),
                html.P(
                    id="subtitle",
                    style={"color": _SUBTEXT, "marginTop": 0, "marginBottom": "18px"},
                ),

                # KPI row
                html.Div(id="kpi-row", style={"display": "flex", "gap": "12px",
                                               "marginBottom": "18px"}),

                # Charts row 1
                html.Div(
                    dcc.Graph(id="chart-main", style={"height": "320px"}),
                    style={"backgroundColor": _CARD_BG, "borderRadius": "8px",
                           "padding": "4px", "marginBottom": "14px"},
                ),

                # Charts row 2
                html.Div(
                    style={"display": "grid",
                           "gridTemplateColumns": "1fr 1fr",
                           "gap": "14px"},
                    children=[
                        html.Div(
                            dcc.Graph(id="chart-error", style={"height": "260px"}),
                            style={"backgroundColor": _CARD_BG,
                                   "borderRadius": "8px", "padding": "4px"},
                        ),
                        html.Div(
                            dcc.Graph(id="chart-hist", style={"height": "260px"}),
                            style={"backgroundColor": _CARD_BG,
                                   "borderRadius": "8px", "padding": "4px"},
                        ),
                    ],
                ),

                # Auto-refresh ticker
                dcc.Interval(id="tick", interval=750, n_intervals=0),
            ],
        )

        # ---- callbacks ----------------------------------------------- #
        @app.callback(
            Output("subtitle",   "children"),
            Output("kpi-row",    "children"),
            Output("chart-main", "figure"),
            Output("chart-error","figure"),
            Output("chart-hist", "figure"),
            Input("tick", "n_intervals"),
        )
        def refresh(_n):
            with self._lock:
                actuals = list(self._actuals)
                preds   = list(self._predictions)
                errors  = list(self._errors)
                ids     = list(self._seq_ids)
                model   = self._model_type

            n = len(actuals)

            # ---- subtitle -------------------------------------------- #
            subtitle = f"Model: {model.upper()}   |   Samples received: {n}"

            # ---- KPI cards ------------------------------------------- #
            def kpi(label, value, color=_TEXT):
                return html.Div(
                    style={"backgroundColor": _CARD_BG, "borderRadius": "8px",
                           "padding": "12px 20px", "minWidth": "140px",
                           "borderLeft": f"4px solid {color}"},
                    children=[
                        html.Div(label,
                                 style={"color": _SUBTEXT, "fontSize": "12px",
                                        "marginBottom": "4px"}),
                        html.Div(value,
                                 style={"color": color, "fontSize": "20px",
                                        "fontWeight": "bold"}),
                    ],
                )

            if n == 0:
                kpis = [kpi("Status", "Waiting for data…", _AMBER)]
                empty_fig = go.Figure(layout=_dark_layout(
                    title=dict(text="Waiting for data…", font=dict(color=_SUBTEXT))
                ))
                return subtitle, kpis, empty_fig, empty_fig, empty_fig

            mae        = float(np.mean(errors))
            latest_act  = actuals[-1]
            latest_pred = preds[-1]

            kpis = [
                kpi("Actual",    f"{latest_act:.4f}",  _BLUE),
                kpi("Predicted", f"{latest_pred:.4f}", _ORANGE),
                kpi("MAE",       f"{mae:.5f}",         _GREEN),
                kpi("Samples",   str(n),                _AMBER),
                kpi("Model",     model.upper(),         _SUBTEXT),
            ]

            # ---- Main chart: actual vs predicted --------------------- #
            fig_main = go.Figure(
                data=[
                    go.Scatter(
                        x=ids, y=actuals,
                        mode="lines", name="Actual",
                        line=dict(color=_BLUE, width=2),
                    ),
                    go.Scatter(
                        x=ids, y=preds,
                        mode="lines", name="Predicted",
                        line=dict(color=_ORANGE, width=2, dash="dash"),
                    ),
                    go.Scatter(
                        x=ids + ids[::-1],
                        y=preds + actuals[::-1],
                        fill="toself",
                        fillcolor="rgba(76,175,80,0.07)",
                        line=dict(color="rgba(0,0,0,0)"),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                ],
                layout=_dark_layout(
                    title=dict(
                        text=f"Actual vs Predicted   (MAE: {mae:.5f})",
                        font=dict(color=_TEXT, size=13),
                    ),
                    xaxis=dict(title="Sequence ID", gridcolor=_GRID,
                               zerolinecolor=_GRID),
                    yaxis=dict(title="Value", gridcolor=_GRID,
                               zerolinecolor=_GRID),
                ),
            )

            # ---- Error over time ------------------------------------- #
            fig_err = go.Figure(
                data=[
                    go.Scatter(
                        x=ids, y=errors,
                        mode="lines", name="|Error|",
                        line=dict(color=_GREEN, width=1.5),
                        fill="tozeroy",
                        fillcolor="rgba(76,175,80,0.1)",
                    ),
                    go.Scatter(
                        x=[ids[0], ids[-1]],
                        y=[mae, mae],
                        mode="lines", name=f"MAE = {mae:.5f}",
                        line=dict(color=_AMBER, width=1.5, dash="dash"),
                    ),
                ],
                layout=_dark_layout(
                    title=dict(text="Absolute Prediction Error",
                               font=dict(color=_TEXT, size=12)),
                    xaxis=dict(title="Sequence ID", gridcolor=_GRID,
                               zerolinecolor=_GRID),
                    yaxis=dict(title="|Error|", gridcolor=_GRID,
                               zerolinecolor=_GRID),
                ),
            )

            # ---- Error histogram ------------------------------------- #
            fig_hist = go.Figure(
                data=[
                    go.Histogram(
                        x=errors,
                        nbinsx=min(30, max(5, n // 5)),
                        marker=dict(
                            color=_GREEN,
                            line=dict(color=_BG, width=0.5),
                        ),
                        opacity=0.8,
                        name="Error",
                    ),
                ],
                layout=_dark_layout(
                    title=dict(text="Error Distribution",
                               font=dict(color=_TEXT, size=12)),
                    xaxis=dict(title="|Error|", gridcolor=_GRID,
                               zerolinecolor=_GRID),
                    yaxis=dict(title="Count", gridcolor=_GRID,
                               zerolinecolor=_GRID),
                    shapes=[dict(
                        type="line",
                        x0=mae, x1=mae, y0=0, y1=1,
                        xref="x", yref="paper",
                        line=dict(color=_AMBER, width=2, dash="dash"),
                    )],
                    annotations=[dict(
                        x=mae, y=1, xref="x", yref="paper",
                        text=f"MAE={mae:.4f}",
                        showarrow=False,
                        font=dict(color=_AMBER, size=10),
                        yanchor="bottom",
                    )],
                ),
            )

            return subtitle, kpis, fig_main, fig_err, fig_hist

        return app

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """
        Start the drain thread and launch the Dash web server.
        Blocks until the process is killed (Ctrl-C).
        """
        drain_thread = threading.Thread(
            target=self._drain_loop,
            daemon=True,
            name="DashboardDrainThread",
        )
        drain_thread.start()

        logger.info(
            "Dashboard running at  http://%s:%d", self.host, self.port
        )
        print(f"\n  Open your browser at  http://{self.host}:{self.port}\n")

        try:
            self._app.run(
                host=self.host,
                port=self.port,
                debug=False,
                use_reloader=False,
            )
        finally:
            self._stop_drain.set()
            drain_thread.join(timeout=3)
            logger.info("Dashboard stopped")
