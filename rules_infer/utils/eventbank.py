# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Tuple

EVENT_TAXONOMY = {
    "RightOfWay": ["intersection_yield", "crosswalk_yield", "roundabout_merge_yield", "bus_stop_merge_yield", "U_turn_yield"],
    "MergeCompete": ["merge_compete", "zipper_merge"],
    "RelativeMotion": ["cut_in", "cut_out", "follow_gap_opening"],
    "ObstacleAvoid": ["double_parking_avoidance", "blocked_intersection_clear", "dooring_avoidance"],
    "PedCycle": ["jaywalking_response", "bike_lane_merge_yield"],
    "Emergency": ["emergency_vehicle_yield", "courtesy_stop"],
    "Congestion": ["congestion_stop", "queue_discharge"],
    "Compliance": ["red_light_stop", "stop_sign_yield", "priority_violation"],
    "Open": ["novel_event"]
}

# ==== 1. Ontology definition (按你原来的分类) ====

PARENT_GROUPS = {
    "right_of_way_exchange": [
        "intersection_yield",
        "crosswalk_yield",
        "roundabout_merge_yield",
        "bus_stop_merge_yield",
        "U_turn_yield",
    ],
    "competition_merge": [
        "merge_compete",
        "zipper_merge",
    ],
    "trajectory_relation": [
        "cut_in",
        "cut_out",
        "follow_gap_opening",
    ],
    "obstacle_avoidance": [
        "double_parking_avoidance",
        "blocked_intersection_clear",
        "dooring_avoidance",
    ],
    "pedestrian_micromobility": [
        "jaywalking_response",
        "bike_lane_merge_yield",
    ],
    "emergency_courtesy": [
        "emergency_vehicle_yield",
        "courtesy_stop",
    ],
    "congestion_state": [
        "congestion_stop",
        "queue_discharge",
    ],
    "compliance_violation": [
        "red_light_stop",
        "stop_sign_yield",
        "priority_violation",
    ],
}

# Flatten closed set labels
CLOSED_SET_LABELS: List[str] = [
    label for group in PARENT_GROUPS.values() for label in group
]

# Novel event special label
NOVEL_EVENT_LABEL = "novel_event"

# Optionally define some mutual exclusivity sets (example, can be tuned)
# These are labels that shouldn't reasonably co-occur as the primary label.
MUTUALLY_EXCLUSIVE_GROUPS = [
    {"congestion_stop", "queue_discharge"},
    {"intersection_yield", "roundabout_merge_yield"},
    {"red_light_stop", "priority_violation"},  # usually choose one as primary
]


def get_parent_group(label: str) -> str:
    """Return the parent group name of a closed-set label, or 'unknown'."""
    for parent, labels in PARENT_GROUPS.items():
        if label in labels:
            return parent
    return "unknown"



# ==== 2. QC / Resolution ====

def resolve_social_event(vlm_json: Dict[str, Any], tau: float = 0.6) -> Dict[str, Any]:
    """
    Decide final social_event based on closed_set_candidates and tau.
    If top candidate < tau -> fall back to novel_event.
    Returns a new dict with an added/overwritten field: 'social_event_resolved'.
    """
    result = dict(vlm_json)  # copy

    candidates = vlm_json.get("closed_set_candidates", [])
    social_event_primary = vlm_json.get("social_event_primary")

    # If model already decided on novel_event, respect it (but still copy to resolved field)
    if social_event_primary == NOVEL_EVENT_LABEL:
        result["social_event_resolved"] = NOVEL_EVENT_LABEL
        return result

    # Parse candidates
    best_label = None
    best_conf = -1.0
    for c in candidates:
        label = c.get("label")
        conf = float(c.get("confidence", 0.0))
        if label in CLOSED_SET_LABELS and conf > best_conf:
            best_label = label
            best_conf = conf

    if best_label is not None and best_conf >= tau:
        # Accept best closed-set label
        result["social_event_resolved"] = best_label
    else:
        # Fall back to novel_event
        result["social_event_resolved"] = NOVEL_EVENT_LABEL

    return result


def check_structure(vlm_json: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Check that required top-level keys exist and that social_event_primary is legal.
    Return (is_ok, list_of_warnings_or_errors).
    """
    required_keys = [
        "direct_observation",
        "causal_inference",
        "social_event_primary",
        "closed_set_candidates",
        "novel_event",
        "event_span",
        "actors",
        "context",
        "evidence",
        "map_refs",
        "confidence_overall",
        "alternatives",
    ]
    messages = []
    ok = True

    for k in required_keys:
        if k not in vlm_json:
            ok = False
            messages.append(f"Missing required key: {k}")

    primary = vlm_json.get("social_event_primary")
    if primary is not None:
        if primary not in CLOSED_SET_LABELS and primary != NOVEL_EVENT_LABEL:
            ok = False
            messages.append(
                f"Invalid social_event_primary: {primary}. Must be one of closed-set labels or '{NOVEL_EVENT_LABEL}'."
            )

    return ok, messages


def check_mutual_exclusion(vlm_json: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Check simple mutual exclusivity rules for the resolved social event vs alternatives.
    This is a light-weight QC; you can extend later.
    """
    messages = []
    ok = True

    primary = vlm_json.get("social_event_resolved", vlm_json.get("social_event_primary"))
    alternatives = vlm_json.get("alternatives", []) or []

    # Build a set of all labels mentioned
    mentioned = {primary}
    mentioned.update(a for a in alternatives if isinstance(a, str))

    for excl_group in MUTUALLY_EXCLUSIVE_GROUPS:
        overlapping = excl_group & mentioned
        if len(overlapping) > 1:
            ok = False
            messages.append(
                f"Mutually exclusive labels appearing together: {sorted(list(overlapping))}"
            )

    return ok, messages


def run_qc_and_resolve(vlm_json: Dict[str, Any], tau: float = 0.6) -> Dict[str, Any]:
    """
    High-level helper:
    1) structural check
    2) resolve social_event via candidates + tau
    3) mutual exclusion check
    Returns:
      {
        "vlm_raw": <original>,
        "vlm_resolved": <with social_event_resolved>,
        "qc": {
          "structure_ok": bool,
          "structure_msgs": [...],
          "mutual_exclusion_ok": bool,
          "mutual_exclusion_msgs": [...]
        }
      }
    """
    struct_ok, struct_msgs = check_structure(vlm_json)
    resolved = resolve_social_event(vlm_json, tau=tau)
    mex_ok, mex_msgs = check_mutual_exclusion(resolved)

    return {
        "vlm_raw": vlm_json,
        "vlm_resolved": resolved,
        "qc": {
            "structure_ok": struct_ok,
            "structure_msgs": struct_msgs,
            "mutual_exclusion_ok": mex_ok,
            "mutual_exclusion_msgs": mex_msgs,
        },
    }
