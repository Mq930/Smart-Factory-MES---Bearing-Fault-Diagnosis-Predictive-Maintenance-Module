"""
Static domain knowledge used to GROUND the LLM's RCA output in real bearing
fault mechanics, rather than letting it invent plausible-sounding but
unverified causes. The RCA prompt in graph.py includes the relevant entry
from FAULT_KNOWLEDGE for the detected class, so the LLM is explaining a
given mechanism rather than guessing one.

Severity mapping (007/014/021) reflects the CWRU dataset's fault diameter
in thousandths of an inch - this is a lab-induced seeded fault, not a
naturally worn one, but the diameter-to-severity ordering is physically
sound and commonly used this way in the bearing-diagnostics literature.
"""

FAULT_KNOWLEDGE = {
    "normal": {
        "mechanical_cause": "No fault signature detected. Bearing operating within normal parameters.",
        "typical_progression": "N/A",
        "recommended_action": "Continue routine monitoring. No maintenance action required.",
    },
    "inner_race": {
        "mechanical_cause": (
            "Localized defect (pit, spall, or crack) on the inner raceway surface. "
            "Produces a periodic impact each time a rolling element passes over the "
            "defect, at the ball-pass frequency inner race (BPFI), which is modulated "
            "by shaft rotation since the inner race rotates with the shaft."
        ),
        "typical_progression": (
            "Inner race defects tend to propagate faster than outer race defects "
            "because the defect passes through the load zone on every rotation. "
            "Commonly escalates from a small pit to spalling and eventual bearing seizure."
        ),
        "recommended_action_by_severity": {
            "007": "Schedule bearing inspection at next planned maintenance window. Increase vibration monitoring frequency.",
            "014": "Schedule bearing replacement within 1-2 maintenance cycles. Monitor for accelerating trend.",
            "021": "Priority bearing replacement recommended. Risk of progression to catastrophic failure if left in service.",
        },
    },
    "outer_race": {
        "mechanical_cause": (
            "Localized defect on the outer raceway surface. Produces a periodic impact "
            "at the ball-pass frequency outer race (BPFO). Unlike inner race defects, "
            "the outer race is typically stationary (fixed to the housing), so the "
            "defect location relative to the load zone stays fixed - the impact "
            "pattern is generally cleaner/more regular in the raw signal."
        ),
        "typical_progression": (
            "Outer race defects tend to propagate more slowly than inner race defects "
            "when the defect sits outside the load zone, but can accelerate rapidly "
            "if positioned within it. Progression is less predictable than inner race."
        ),
        "recommended_action_by_severity": {
            "007": "Schedule bearing inspection at next planned maintenance window. Continue monitoring.",
            "014": "Schedule bearing replacement within 1-2 maintenance cycles.",
            "021": "Priority bearing replacement recommended. Inspect for housing/alignment issues that may have contributed.",
        },
    },
    "ball": {
        "mechanical_cause": (
            "Localized defect on one or more rolling elements (balls). Produces an "
            "impact at the ball spin frequency (BSF) each time the defect contacts "
            "the raceway. Often harder to detect early since the defect's orientation "
            "relative to the load zone varies as the ball rotates and orbits."
        ),
        "typical_progression": (
            "Ball defects can cause secondary damage to both raceways if left "
            "unaddressed, since a damaged ball repeatedly impacts both the inner "
            "and outer race surfaces. Progression can be harder to trend than "
            "race defects due to the varying contact geometry."
        ),
        "recommended_action_by_severity": {
            "007": "Schedule bearing inspection. Ball defects can be harder to detect early - consider closer monitoring interval.",
            "014": "Schedule bearing replacement within 1-2 maintenance cycles. Inspect races for secondary damage.",
            "021": "Priority bearing replacement recommended. Inspect both raceways for secondary damage from ball impacts.",
        },
    },
}


def get_fault_info(class_name: str) -> dict:
    """
    class_name: one of CLASS_NAMES from dataset.py, e.g. "inner_race_014".
    Returns a dict with mechanical_cause, typical_progression, and
    recommended_action (severity-specific if applicable).
    """
    if class_name == "normal":
        return FAULT_KNOWLEDGE["normal"]

    parts = class_name.rsplit("_", 1)
    if len(parts) != 2 or parts[1] not in ("007", "014", "021"):
        raise ValueError(f"Unrecognized class name format: {class_name}")
    fault_type, severity = parts

    if fault_type not in FAULT_KNOWLEDGE:
        raise ValueError(f"Unknown fault type: {fault_type}")

    info = dict(FAULT_KNOWLEDGE[fault_type])  # shallow copy
    info["severity"] = severity
    info["recommended_action"] = info["recommended_action_by_severity"][severity]
    return info
