"""
LangGraph MOM (Manufacturing Operations Management) agent.

Graph flow:
    ingest -> aggregate_trend -> route_severity -> [rca_report | END]

    - ingest: takes a batch of raw windows + the loaded classifier tool,
      returns per-window predictions.
    - aggregate_trend: summarizes the batch into a single trend verdict -
      majority class, confidence stats, and whether the batch shows a
      CONSISTENT fault pattern vs. a NOISY/mixed one. This is what makes
      this a "batch of windows -> aggregated trend" agent rather than
      reacting to any single noisy window.
    - route_severity: maps the aggregated trend to an alert level
      (normal / watch / alert / critical) using simple, inspectable rules -
      not an LLM call, so alert routing is deterministic and auditable.
    - rca_report: ONLY runs if a fault was detected. Calls the Groq LLM to
      write a structured RCA report, but the report is grounded in
      fault_knowledge.py's mechanical facts and the actual aggregated
      classifier stats - the LLM's job is to phrase and structure a report
      from given facts, not to invent the diagnosis itself.

State is intentionally flat and serializable (no numpy arrays in the
TypedDict) so it can be persisted/logged/sent over an API later.
"""

import os
import sys
from collections import Counter
from typing import List, Optional, TypedDict

import numpy as np
from langgraph.graph import StateGraph, END

sys.path.insert(0, os.path.dirname(__file__))
from classifier_tool import BearingClassifierTool, WindowPrediction  # noqa: E402
from fault_knowledge import get_fault_info                            # noqa: E402

try:
    from groq import Groq
except ImportError:
    Groq = None  # allows graph.py to be imported/tested without groq installed,
                 # as long as the rca_report node is never actually invoked


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class MOMState(TypedDict, total=False):
    # --- input ---
    raw_windows: object          # np.ndarray (N, window_size) - set by caller before invoke

    # --- ingest node output ---
    window_predictions: List[dict]   # serialized WindowPrediction (class, confidence, probs)

    # --- aggregate_trend node output ---
    majority_class: str
    majority_fraction: float         # fraction of windows agreeing with majority_class
    mean_confidence: float
    is_consistent: bool              # majority_fraction above CONSISTENCY_THRESHOLD
    class_distribution: dict         # class_name -> count across the batch

    # --- route_severity node output ---
    alert_level: str                 # "normal" | "watch" | "alert" | "critical"

    # --- rca_report node output ---
    rca_report: Optional[str]        # None if alert_level == "normal"


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

CONSISTENCY_THRESHOLD = 0.6   # majority class must cover >=60% of windows to be "consistent"
HIGH_CONFIDENCE_THRESHOLD = 0.9
SEVERE_FAULT_SUFFIXES = ("014", "021")  # anything but the mildest (007) severity


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def make_ingest_node(classifier: BearingClassifierTool):
    """Factory so the node closes over a pre-loaded classifier instance
    (loading the checkpoint per-call would be far too slow)."""

    def ingest(state: MOMState) -> MOMState:
        windows = state["raw_windows"]
        predictions: List[WindowPrediction] = classifier.classify_batch(windows)

        serialized = [
            {
                "window_index": p.window_index,
                "predicted_class": p.predicted_class,
                "confidence": p.confidence,
                "all_probs": p.all_probs,
            }
            for p in predictions
        ]
        return {"window_predictions": serialized}

    return ingest


def aggregate_trend(state: MOMState) -> MOMState:
    preds = state["window_predictions"]
    classes = [p["predicted_class"] for p in preds]
    confidences = [p["confidence"] for p in preds]

    counts = Counter(classes)
    majority_class, majority_count = counts.most_common(1)[0]
    majority_fraction = majority_count / len(classes)
    mean_confidence = float(np.mean(confidences))

    return {
        "majority_class": majority_class,
        "majority_fraction": majority_fraction,
        "mean_confidence": mean_confidence,
        "is_consistent": majority_fraction >= CONSISTENCY_THRESHOLD,
        "class_distribution": dict(counts),
    }


def route_severity(state: MOMState) -> MOMState:
    """
    Deterministic, inspectable routing rules - NOT an LLM decision, so the
    alert level is always auditable/reproducible from the aggregated stats.

        normal:   majority class is "normal"
        watch:    fault detected, but inconsistent across the batch OR low confidence
                   (i.e. might be a transient/borderline reading, worth watching)
        alert:    fault detected consistently, high confidence, mild severity (007)
        critical: fault detected consistently, high confidence, severe (014/021)
    """
    majority_class = state["majority_class"]

    if majority_class == "normal":
        alert_level = "normal"
    elif not state["is_consistent"] or state["mean_confidence"] < HIGH_CONFIDENCE_THRESHOLD:
        alert_level = "watch"
    elif majority_class.endswith(SEVERE_FAULT_SUFFIXES):
        alert_level = "critical"
    else:
        alert_level = "alert"

    return {"alert_level": alert_level}


def make_rca_node(groq_api_key: str = None, model: str = "llama-3.3-70b-versatile"):
    def rca_report(state: MOMState) -> MOMState:
        if state["alert_level"] == "normal":
            return {"rca_report": None}

        if Groq is None:
            raise ImportError("groq package not installed. Run: pip install groq")

        client = Groq(api_key=groq_api_key or os.environ.get("GROQ_API_KEY"))

        fault_info = get_fault_info(state["majority_class"])

        # Ground the prompt in the ACTUAL aggregated stats and known fault
        # mechanics - the LLM structures/phrases the report, it does not
        # invent the diagnosis, severity, or recommended action.
        prompt = f"""You are a predictive-maintenance assistant for a bearing fault diagnosis MES module.
Write a structured RCA (Root Cause Analysis) report using ONLY the facts given below. Do not invent statistics, mechanical details, or recommendations beyond what's provided - your job is to structure and clearly phrase this information for a shop-floor maintenance team, not to add new claims.

DETECTED CONDITION:
- Majority classified fault: {state['majority_class']}
- Alert level: {state['alert_level']}
- Fraction of monitoring window agreeing with this classification: {state['majority_fraction']:.0%}
- Mean model confidence: {state['mean_confidence']:.1%}
- Full class distribution across the monitoring window: {state['class_distribution']}

KNOWN MECHANICAL FACTS (from bearing fault knowledge base):
- Mechanical cause: {fault_info['mechanical_cause']}
- Typical progression: {fault_info['typical_progression']}
- Recommended action: {fault_info['recommended_action']}

Write the report with these sections: SEVERITY, MECHANICAL CAUSE, TREND OBSERVATION (referencing the consistency/confidence stats above), RECOMMENDED ACTION. Keep it concise and factual, appropriate for a maintenance engineer to act on quickly."""

        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a precise, factual predictive-maintenance report writer. Never invent data not given to you."},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.2,  # low temperature - this is a factual report, not creative writing
        )
        report_text = completion.choices[0].message.content
        return {"rca_report": report_text}

    return rca_report


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_mom_graph(checkpoint_path: str, groq_api_key: str = None,
                     groq_model: str = "llama-3.3-70b-versatile"):
    """
    Returns (compiled_graph, classifier_tool). The classifier_tool is
    returned too since query_cli.py needs direct access to it for ad-hoc
    classification outside the graph flow (e.g. answering "what's the
    current reading" without re-running the full pipeline).
    """
    classifier = BearingClassifierTool(checkpoint_path)

    graph = StateGraph(MOMState)
    graph.add_node("ingest", make_ingest_node(classifier))
    graph.add_node("aggregate_trend", aggregate_trend)
    graph.add_node("route_severity", route_severity)
    graph.add_node("rca_report", make_rca_node(groq_api_key, groq_model))

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "aggregate_trend")
    graph.add_edge("aggregate_trend", "route_severity")

    # rca_report always runs after route_severity, but it internally
    # short-circuits to rca_report=None when alert_level == "normal" -
    # kept as a single node (not a conditional edge to END) so the state
    # shape stays consistent for every run, which simplifies query_cli.py.
    graph.add_edge("route_severity", "rca_report")
    graph.add_edge("rca_report", END)

    compiled = graph.compile()
    return compiled, classifier


if __name__ == "__main__":
    # Quick structural smoke test - does NOT call the Groq API (would
    # require a real key), just verifies the graph compiles and the
    # deterministic nodes (ingest/aggregate/route) run correctly.
    import numpy as np

    print("Building graph (loads checkpoint)...")
    print("NOTE: run this from mom_agent/ with a valid ../checkpoints/best_model.pt")
