"""
Simple CLI loop simulating the shop-floor natural-language query interface.

Flow:
    1. On startup, runs the MOM graph once on a batch of windows (from a
       real CWRU recording, standing in for a live sensor feed) to produce
       an initial diagnosis + RCA report.
    2. Drops into an interactive loop where an operator can ask questions
       in plain English about the current diagnosis. Questions are answered
       by a Groq LLM call, grounded in the current agent state (same
       pattern as the RCA node - the LLM explains/summarizes given facts,
       it does not have access to run new classifications or invent data).
    3. Special commands: "rerun" to pull a new batch and re-diagnose,
       "quit"/"exit" to stop.

Usage:
    python query_cli.py --checkpoint ../checkpoints/best_model.pt \
        --raw_dir ../data/raw --class_name inner_race_014 --load 0
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from classifier_tool import load_windows_from_recording  # noqa: E402
from graph import build_mom_graph                         # noqa: E402

try:
    from groq import Groq
except ImportError:
    Groq = None


def run_diagnosis(compiled_graph, raw_dir: str, class_name: str, load: int,
                   n_windows: int, start_at: int) -> dict:
    windows = load_windows_from_recording(
        raw_dir, class_name, load, n_windows=n_windows, start_at=start_at
    )
    initial_state = {"raw_windows": windows}
    final_state = compiled_graph.invoke(initial_state)
    return final_state


def print_diagnosis_summary(state: dict):
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)
    print(f"  Majority class:      {state['majority_class']}")
    print(f"  Alert level:         {state['alert_level'].upper()}")
    print(f"  Batch consistency:   {state['majority_fraction']:.0%} of windows agree")
    print(f"  Mean confidence:     {state['mean_confidence']:.1%}")
    print(f"  Class distribution:  {state['class_distribution']}")
    if state.get("rca_report"):
        print("\n--- RCA REPORT ---")
        print(state["rca_report"])
    print("=" * 60 + "\n")


def answer_query(question: str, state: dict, groq_api_key: str = None,
                  model: str = "llama-3.3-70b-versatile") -> str:
    """
    Answers an operator's natural-language question using ONLY the current
    diagnosis state as context - same grounding principle as the RCA node.
    """
    if Groq is None:
        return "[Error: groq package not installed. Run: pip install groq]"

    client = Groq(api_key=groq_api_key or os.environ.get("GROQ_API_KEY"))

    context = f"""Current bearing diagnosis state (from the MES predictive-maintenance system):
- Majority classified condition: {state['majority_class']}
- Alert level: {state['alert_level']}
- Batch consistency: {state['majority_fraction']:.0%} of monitored windows agree with this classification
- Mean model confidence: {state['mean_confidence']:.1%}
- Full class distribution: {state['class_distribution']}
- RCA report on file: {state.get('rca_report') or '(none - no fault detected)'}
"""

    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": (
                "You are a shop-floor assistant for a bearing predictive-maintenance system. "
                "Answer the operator's question using ONLY the diagnosis state provided below. "
                "If the question asks for information not present in the state, say so clearly "
                "rather than guessing or inventing details."
            )},
            {"role": "user", "content": f"{context}\n\nOperator question: {question}"},
        ],
        model=model,
        temperature=0.2,
    )
    return completion.choices[0].message.content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="../checkpoints/best_model.pt")
    ap.add_argument("--raw_dir", default="../data/raw")
    ap.add_argument("--class_name", default="inner_race_014",
                     help="which recording to pull the initial batch from (simulated sensor feed)")
    ap.add_argument("--load", type=int, default=0)
    ap.add_argument("--n_windows", type=int, default=10,
                     help="how many consecutive windows form one 'batch' / trend snapshot")
    ap.add_argument("--start_at", type=int, default=0,
                     help="starting window index within the recording")
    ap.add_argument("--groq_model", default="llama-3.3-70b-versatile")
    args = ap.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY not set in environment. RCA reports and "
              "queries will fail until you set it:\n"
              "  export GROQ_API_KEY=your_key_here  (Linux/Mac)\n"
              "  set GROQ_API_KEY=your_key_here      (Windows cmd)\n")

    print("Loading model checkpoint and compiling agent graph...")
    compiled_graph, classifier = build_mom_graph(args.checkpoint, groq_model=args.groq_model)
    print(f"Ready. Class names: {classifier.class_names}\n")

    print(f"Running initial diagnosis on {args.n_windows} windows from "
          f"'{args.class_name}' load={args.load} (simulated sensor batch)...")
    state = run_diagnosis(compiled_graph, args.raw_dir, args.class_name,
                           args.load, args.n_windows, args.start_at)
    print_diagnosis_summary(state)

    print("Shop-floor query interface ready. Ask a question, or:")
    print("  'rerun <class_name> <load>' - pull a new batch and re-diagnose")
    print("  'quit' / 'exit' - stop\n")

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Exiting.")
            break

        if user_input.lower().startswith("rerun"):
            parts = user_input.split()
            new_class = parts[1] if len(parts) > 1 else args.class_name
            new_load = int(parts[2]) if len(parts) > 2 else args.load
            try:
                state = run_diagnosis(compiled_graph, args.raw_dir, new_class,
                                       new_load, args.n_windows, args.start_at)
                print_diagnosis_summary(state)
            except ValueError as e:
                print(f"Error: {e}\n")
            continue

        answer = answer_query(user_input, state, model=args.groq_model)
        print(f"\n{answer}\n")


if __name__ == "__main__":
    main()
