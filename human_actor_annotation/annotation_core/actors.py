from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ACTOR_LIST_KEYS: Tuple[str, ...] = (
    "actors",
    "boxes",
    "bounding_boxes",
    "objects",
    "detections",
    "agents",
)


def read_json_auto(path: Path) -> Any:
    if path.suffix == ".gz" or path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_actor_list(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []

    for key in ACTOR_LIST_KEYS:
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    # Some files store one dictionary per actor under numeric/string keys.
    dict_values = [value for value in data.values() if isinstance(value, dict)]
    if dict_values and len(dict_values) == len(data):
        return dict_values

    # Last-resort shallow recursive search.
    for value in data.values():
        if isinstance(value, dict):
            result = _find_actor_list(value)
            if result:
                return result
    return []


def _first_value(actor: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in actor and actor[key] is not None:
            return actor[key]
    return default


def _safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _vector_xy(value: Any) -> Tuple[Optional[float], Optional[float]]:
    if isinstance(value, dict):
        x = _safe_float(_first_value(value, ["x", "x_m", "forward"]))
        y = _safe_float(_first_value(value, ["y", "y_m", "right"]))
        return x, y
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2:
        return _safe_float(value[0]), _safe_float(value[1])
    return None, None


def _extract_xy(actor: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    x = _safe_float(_first_value(actor, ["x_m", "relative_x", "forward_m"]))
    y = _safe_float(_first_value(actor, ["y_m", "relative_y", "right_m"]))
    if x is not None and y is not None:
        return x, y

    for key in ["position", "location", "relative_position_xyz", "center", "translation"]:
        if key in actor:
            x, y = _vector_xy(actor[key])
            if x is not None and y is not None:
                return x, y

    # CARLA transforms/matrices occasionally contain the relative translation.
    matrix = actor.get("matrix") or actor.get("transform")
    if isinstance(matrix, list):
        arr = np.asarray(matrix)
        if arr.shape == (4, 4):
            return _safe_float(arr[0, 3]), _safe_float(arr[1, 3])
    if isinstance(matrix, dict):
        for key in ["location", "position", "translation"]:
            if key in matrix:
                x, y = _vector_xy(matrix[key])
                if x is not None and y is not None:
                    return x, y

    return None, None


def _normalize_class(actor: Dict[str, Any]) -> str:
    raw = str(
        _first_value(
            actor,
            ["class", "type", "category", "label", "object_type", "semantic_class"],
            "unknown",
        )
    ).lower()
    raw = raw.replace("vehicle.", "vehicle_").replace("walker.", "walker_")

    if "pedestrian" in raw or "walker" in raw or "person" in raw:
        return "pedestrian"
    if "bicycle" in raw or "cyclist" in raw or "bike" in raw:
        return "bicycle"
    if "motorcycle" in raw or "motorbike" in raw:
        return "motorcycle"
    if "truck" in raw:
        return "truck"
    if "bus" in raw:
        return "bus"
    if "vehicle" in raw or "car" in raw:
        return "vehicle"
    if "cone" in raw:
        return "traffic_cone"
    if "static" in raw or "obstacle" in raw or "warning" in raw:
        return "static_obstacle"
    return raw


def _actor_id(actor: Dict[str, Any], fallback_index: int) -> str:
    value = _first_value(actor, ["id", "actor_id", "track_id", "instance_id", "token"])
    if value is not None:
        return str(value)
    x, y = _extract_xy(actor)
    payload = f"{_normalize_class(actor)}:{x}:{y}:{fallback_index}".encode("utf-8")
    return "fallback_" + hashlib.sha1(payload).hexdigest()[:10]


def normalize_actor(actor: Dict[str, Any], fallback_index: int) -> Dict[str, Any]:
    x, y = _extract_xy(actor)
    distance = _safe_float(_first_value(actor, ["distance", "distance_m", "range_m"]))
    if distance is None and x is not None and y is not None:
        distance = float(math.hypot(x, y))

    speed = _safe_float(_first_value(actor, ["speed", "speed_mps", "velocity_mps"]))
    extent = _first_value(actor, ["extent", "size", "dimensions"])

    role_name = str(_first_value(actor, ["role_name", "role", "name"], "")).lower()
    normalized = {
        "actor_id": _actor_id(actor, fallback_index),
        "class": _normalize_class(actor),
        "raw_class": str(_first_value(actor, ["class", "type", "category", "label"], "unknown")),
        "x_m": x,
        "y_m": y,
        "distance_m": distance,
        "speed_mps": speed,
        "role_name": role_name,
        "extent": extent,
        "raw": actor,
    }
    return normalized


def load_normalized_actors(boxes_path: Path) -> List[Dict[str, Any]]:
    data = read_json_auto(boxes_path)
    actors = _find_actor_list(data)
    return [normalize_actor(actor, index) for index, actor in enumerate(actors)]


def _class_matches(class_name: str, patterns: Iterable[str]) -> bool:
    class_name = class_name.lower()
    return any(pattern.lower() in class_name for pattern in patterns)


def select_candidates(
    actors: List[Dict[str, Any]],
    include_classes: Sequence[str],
    exclude_classes: Sequence[str],
    max_distance_m: float,
    min_forward_m: float,
    max_candidates: int,
    sample_id: str,
    random_seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    stats = {
        "total": len(actors),
        "excluded_ego": 0,
        "excluded_class": 0,
        "excluded_distance": 0,
        "excluded_position": 0,
        "missing_position": 0,
        "truncated": 0,
    }

    selected: List[Dict[str, Any]] = []
    for actor in actors:
        cls = actor["class"].lower()
        raw_cls = str(actor.get("raw_class", "")).lower()
        role = actor.get("role_name", "").lower()
        if (
            _class_matches(cls, exclude_classes)
            or _class_matches(raw_cls, exclude_classes)
            or role in {"hero", "ego", "autopilot"}
        ):
            stats["excluded_ego"] += 1
            continue
        if include_classes and not _class_matches(cls, include_classes):
            stats["excluded_class"] += 1
            continue

        x = actor.get("x_m")
        distance = actor.get("distance_m")
        if x is None or actor.get("y_m") is None:
            stats["missing_position"] += 1
            continue
        if x < min_forward_m:
            stats["excluded_position"] += 1
            continue
        if distance is not None and distance > max_distance_m:
            stats["excluded_distance"] += 1
            continue
        selected.append(dict(actor))

    # Keep the nearest actors before truncation, but randomize displayed labels later.
    selected.sort(
        key=lambda a: (
            float(a.get("distance_m")) if a.get("distance_m") is not None else float("inf"),
            str(a.get("actor_id")),
        )
    )
    if len(selected) > max_candidates:
        stats["truncated"] = len(selected) - max_candidates
        selected = selected[:max_candidates]

    # Stable per-sample randomization prevents A1 from always meaning "nearest".
    seed_payload = f"{random_seed}:{sample_id}".encode("utf-8")
    per_sample_seed = int(hashlib.sha1(seed_payload).hexdigest()[:12], 16)
    rng = random.Random(per_sample_seed)
    rng.shuffle(selected)

    for index, actor in enumerate(selected, start=1):
        actor["display_label"] = f"A{index}"
    return selected, stats
