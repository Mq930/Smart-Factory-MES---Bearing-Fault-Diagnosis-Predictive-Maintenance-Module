"""
FastAPI backend for the Smart Factory MES React dashboard.

Wraps the existing model/Grad-CAM/MOM-agent pipeline (src/ and mom_agent/)
behind HTTP endpoints. The heavy objects (trained checkpoint, MOM graph)
are loaded ONCE at server startup via the lifespan context manager, not
per-request - loading a checkpoint on every API call would be far too slow.

Run with:
    uvicorn main:app --reload --port 8000

Endpoints:
    GET  /recordings                          - list available (class, load) combos
    POST /diagnose                             - run the full MOM agent graph on a batch
    GET  /telemetry/{class_name}/{load}        - raw windows + Grad-CAM saliency (no LLM)
    POST /query                                - ask a question about the last diagnosis
    GET  /health                               - basic liveness check
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mom_agent"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from classifier_tool import load_windows_from_recording  # noqa: E402
from graph import build_mom_graph                         # noqa: E402
from query_cli import answer_query                        # noqa: E402
from dataset import build_recordings                      # noqa: E402


# ---------------------------------------------------------------------------
# Config - adjust these paths/values if your checkpoint or data live elsewhere
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = os.environ.get(
    "BEARING_CHECKPOINT", os.path.join(os.path.dirname(__file__), "..", "checkpoints", "best_model.pt")
)
RAW_DATA_DIR = os.environ.get(
    "BEARING_RAW_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "raw")
)
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


# ---------------------------------------------------------------------------
# App state - populated once at startup, shared across requests.
# Simple module-level dict since this is a single-process demo server;
# a production deployment would use app.state or a proper DI container.
# ---------------------------------------------------------------------------

app_state = {
    "compiled_graph": None,
    "classifier": None,
    "last_diagnosis_state": None,  # most recent MOM agent run, for /query to reference
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading checkpoint from {CHECKPOINT_PATH} ...")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. Train a model first "
            f"(see src/train.py) or set the BEARING_CHECKPOINT env var."
        )
    compiled_graph, classifier = build_mom_graph(CHECKPOINT_PATH, groq_model=GROQ_MODEL)
    app_state["compiled_graph"] = compiled_graph
    app_state["classifier"] = classifier
    print("Model + MOM agent graph loaded. Ready to serve requests.")

    yield  # server runs here

    print("Shutting down.")
    app_state.clear()


app = FastAPI(title="Bearing Fault Diagnosis MES API", lifespan=lifespan)

# CORS: allow the React dev server (Vite default port 5173, CRA default 3000)
# to call this API from a different origin. Tighten allow_origins for
# production deployments.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class RecordingInfo(BaseModel):
    class_name: str
    load: int
    n_windows_available: int


class DiagnoseRequest(BaseModel):
    class_name: str
    load: int
    n_windows: int = 10
    start_at: int = 0


class WindowPredictionOut(BaseModel):
    window_index: int
    predicted_class: str
    confidence: float
    all_probs: dict


class DiagnoseResponse(BaseModel):
    majority_class: str
    majority_fraction: float
    mean_confidence: float
    is_consistent: bool
    class_distribution: dict
    alert_level: str
    rca_report: Optional[str]
    window_predictions: List[WindowPredictionOut]


class TelemetryWindowOut(BaseModel):
    window_index: int
    signal: List[float]           # raw (un-normalized) waveform, for plotting
    cam: List[float]              # Grad-CAM saliency, same length as signal
    predicted_class: str
    confidence: float


class TelemetryResponse(BaseModel):
    class_name: str
    load: int
    windows: List[TelemetryWindowOut]


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app_state["classifier"] is not None}


@app.get("/recordings", response_model=List[RecordingInfo])
def list_recordings():
    """
    Lists every (class_name, load) recording available in RAW_DATA_DIR, so
    the frontend can populate a selector instead of hardcoding options.
    """
    try:
        recordings = build_recordings(RAW_DATA_DIR)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return [
        RecordingInfo(class_name=r.class_name, load=r.load, n_windows_available=len(r.windows))
        for r in recordings
    ]


@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose(req: DiagnoseRequest):
    """
    Runs the full MOM agent graph (ingest -> aggregate_trend ->
    route_severity -> rca_report) on a batch of consecutive windows pulled
    from a real recording (simulating a rolling sensor feed - see
    classifier_tool.load_windows_from_recording's docstring for how this
    would be swapped for a live ingestion source in production).

    Also stores the result in app_state so /query can reference it.
    """
    try:
        windows = load_windows_from_recording(
            RAW_DATA_DIR, req.class_name, req.load,
            n_windows=req.n_windows, start_at=req.start_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    compiled_graph = app_state["compiled_graph"]
    final_state = compiled_graph.invoke({"raw_windows": windows})

    app_state["last_diagnosis_state"] = final_state

    return DiagnoseResponse(
        majority_class=final_state["majority_class"],
        majority_fraction=final_state["majority_fraction"],
        mean_confidence=final_state["mean_confidence"],
        is_consistent=final_state["is_consistent"],
        class_distribution=final_state["class_distribution"],
        alert_level=final_state["alert_level"],
        rca_report=final_state.get("rca_report"),
        window_predictions=[
            WindowPredictionOut(**p) for p in final_state["window_predictions"]
        ],
    )


@app.get("/telemetry/{class_name}/{load}", response_model=TelemetryResponse)
def telemetry(class_name: str, load: int, n_windows: int = 10, start_at: int = 0):
    """
    Returns raw waveforms + Grad-CAM saliency for visualization, WITHOUT
    running the full MOM agent graph (no LLM call, no severity routing) -
    this is the lightweight endpoint the dashboard polls to draw the live
    telemetry + saliency overlay chart.
    """
    try:
        windows = load_windows_from_recording(
            RAW_DATA_DIR, class_name, load, n_windows=n_windows, start_at=start_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    classifier = app_state["classifier"]
    predictions = classifier.classify_batch(windows)

    windows_out = []
    for raw_signal, pred in zip(windows, predictions):
        windows_out.append(TelemetryWindowOut(
            window_index=pred.window_index,
            signal=raw_signal.tolist(),
            cam=pred.cam.tolist(),
            predicted_class=pred.predicted_class,
            confidence=pred.confidence,
        ))

    return TelemetryResponse(class_name=class_name, load=load, windows=windows_out)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Answers a natural-language question grounded in the most recent
    /diagnose result. Returns 400 if /diagnose hasn't been called yet in
    this server session - there's nothing to ground the answer in.
    """
    state = app_state["last_diagnosis_state"]
    if state is None:
        raise HTTPException(
            status_code=400,
            detail="No diagnosis has been run yet. Call /diagnose first.",
        )

    answer = answer_query(req.question, state, model=GROQ_MODEL)
    return QueryResponse(answer=answer)
