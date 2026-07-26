from __future__ import annotations

import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    mode = "rt" if path.name.endswith(".gz") else "r"
    with opener(path, mode, encoding="utf-8") as f:  # type: ignore[arg-type]
        text = f.read()
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped.startswith("["):
        data = json.loads(text)
        return [item for item in data if isinstance(item, dict)]
    if stripped.startswith("{") and "\n" not in stripped.rstrip("\n"):
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ["records", "samples", "predictions", "annotations"]:
                if isinstance(data.get(key), list):
                    return [item for item in data[key] if isinstance(item, dict)]
            return [data]
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            item = json.loads(line)
            if isinstance(item, dict):
                records.append(item)
    return records


def read_annotation_dir(annotation_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in sorted(annotation_dir.glob("annotations_*.jsonl")):
        # Keep only the latest revision for each annotator/sample in each file.
        latest: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for record in read_json_or_jsonl(path):
            key = (str(record.get("annotator_id", path.stem)), str(record.get("sample_id", "")))
            if key[1]:
                latest[key] = record
        records.extend(latest.values())
    return records


def annotation_category(record: Dict[str, Any]) -> str:
    judgement = str(record.get("scene_judgement", ""))
    labels = [str(x) for x in record.get("selected_display_labels", []) if str(x)]
    if record.get("is_skipped", False):
        return "SKIP"
    if judgement == "无关键对象":
        return "NONE"
    if judgement == "无法判断":
        return "UNCERTAIN"
    if judgement == "多个对象共同影响":
        return "MULTI:" + "+".join(sorted(labels))
    if labels:
        return labels[0]
    return "INVALID"


def pairwise_agreement(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_sample[str(record.get("sample_id", ""))].append(record)

    exact_total = 0
    exact_match = 0
    dominant_total = 0
    dominant_match = 0
    for sample_records in by_sample.values():
        for a, b in combinations(sample_records, 2):
            exact_total += 1
            if annotation_category(a) == annotation_category(b):
                exact_match += 1

            a_ids = set(str(x) for x in a.get("selected_actor_ids", []))
            b_ids = set(str(x) for x in b.get("selected_actor_ids", []))
            if len(a_ids) == 1 and len(b_ids) == 1:
                dominant_total += 1
                if a_ids == b_ids:
                    dominant_match += 1

    return {
        "pairwise_exact_agreement": exact_match / exact_total if exact_total else float("nan"),
        "pairwise_dominant_actor_agreement": dominant_match / dominant_total if dominant_total else float("nan"),
        "pair_count": exact_total,
        "dominant_pair_count": dominant_total,
    }


def fleiss_kappa(records: Sequence[Dict[str, Any]]) -> float:
    by_sample: Dict[str, List[str]] = defaultdict(list)
    for record in records:
        category = annotation_category(record)
        if category not in {"SKIP", "INVALID"}:
            by_sample[str(record.get("sample_id", ""))].append(category)

    # Classical Fleiss kappa requires a constant number of raters per item.
    counts_by_n = Counter(len(values) for values in by_sample.values() if len(values) >= 2)
    if not counts_by_n:
        return float("nan")
    n_raters = counts_by_n.most_common(1)[0][0]
    items = [values for values in by_sample.values() if len(values) == n_raters]
    if not items or n_raters < 2:
        return float("nan")

    categories = sorted({category for values in items for category in values})
    category_index = {category: index for index, category in enumerate(categories)}
    matrix = []
    for values in items:
        row = [0] * len(categories)
        for value in values:
            row[category_index[value]] += 1
        matrix.append(row)

    n_items = len(matrix)
    p_j = [sum(row[j] for row in matrix) / (n_items * n_raters) for j in range(len(categories))]
    p_e = sum(p * p for p in p_j)
    p_i = [
        (sum(count * count for count in row) - n_raters) / (n_raters * (n_raters - 1))
        for row in matrix
    ]
    p_bar = sum(p_i) / len(p_i)
    if abs(1.0 - p_e) < 1e-12:
        return float("nan")
    return (p_bar - p_e) / (1.0 - p_e)


def build_consensus(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not record.get("is_skipped", False):
            by_sample[str(record.get("sample_id", ""))].append(record)

    consensus: List[Dict[str, Any]] = []
    for sample_id, sample_records in sorted(by_sample.items()):
        categories = [annotation_category(record) for record in sample_records]
        category_counts = Counter(categories)
        winning_category, winning_votes = category_counts.most_common(1)[0]
        tied = sum(1 for count in category_counts.values() if count == winning_votes) > 1

        actor_vote_counter: Counter[str] = Counter()
        for record in sample_records:
            for actor_id in record.get("selected_actor_ids", []):
                actor_vote_counter[str(actor_id)] += 1
        top_actor_id = None
        top_actor_votes = 0
        actor_tied = False
        if actor_vote_counter:
            top_actor_id, top_actor_votes = actor_vote_counter.most_common(1)[0]
            actor_tied = sum(1 for count in actor_vote_counter.values() if count == top_actor_votes) > 1

        confidences = [
            float(record.get("confidence"))
            for record in sample_records
            if record.get("confidence") is not None
        ]
        consensus.append(
            {
                "sample_id": sample_id,
                "n_annotators": len(sample_records),
                "consensus_category": winning_category,
                "category_votes": winning_votes,
                "category_vote_ratio": winning_votes / len(sample_records),
                "category_tied": tied,
                "consensus_actor_id": None if actor_tied else top_actor_id,
                "actor_votes": top_actor_votes,
                "actor_vote_ratio": top_actor_votes / len(sample_records) if top_actor_id else 0.0,
                "actor_tied": actor_tied,
                "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
                "needs_adjudication": bool(tied or actor_tied or winning_votes <= len(sample_records) / 2),
            }
        )
    return consensus


def _nested_get(data: Dict[str, Any], dotted: str) -> Any:
    current: Any = data
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def prediction_actor_id(record: Dict[str, Any]) -> Optional[str]:
    for key in [
        "selected_actor_id",
        "critical_actor_id",
        "causal_actor_id",
        "actor_id",
        "causal_object.id",
        "critical_actor.id",
        "label.causal_object.id",
        "result.causal_object.id",
    ]:
        value = _nested_get(record, key)
        if value is not None:
            return str(value)
    return None


def compare_predictions(
    consensus: Sequence[Dict[str, Any]],
    prediction_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    predictions: Dict[str, Optional[str]] = {}
    for record in prediction_records:
        sample_id = str(record.get("sample_id", ""))
        if sample_id:
            predictions[sample_id] = prediction_actor_id(record)

    eligible = 0
    correct = 0
    missing_prediction = 0
    disagreements: List[Dict[str, Any]] = []
    for item in consensus:
        human_actor = item.get("consensus_actor_id")
        if not human_actor or item.get("needs_adjudication", False):
            continue
        eligible += 1
        predicted = predictions.get(str(item["sample_id"]))
        if predicted is None:
            missing_prediction += 1
            disagreements.append({**item, "predicted_actor_id": None, "reason": "missing_prediction"})
        elif str(predicted) == str(human_actor):
            correct += 1
        else:
            disagreements.append({**item, "predicted_actor_id": predicted, "reason": "actor_mismatch"})

    return {
        "eligible_consensus_samples": eligible,
        "correct": correct,
        "top1_accuracy": correct / eligible if eligible else float("nan"),
        "missing_predictions": missing_prediction,
        "disagreements": disagreements,
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serialized = {}
            for key, value in row.items():
                if isinstance(value, (list, dict)):
                    serialized[key] = json.dumps(value, ensure_ascii=False)
                else:
                    serialized[key] = value
            writer.writerow(serialized)
