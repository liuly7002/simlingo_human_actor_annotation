from __future__ import annotations

import csv
import fcntl
import json
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


ANNOTATION_FIELDS = [
    "schema_version",
    "annotator_id",
    "sample_id",
    "route_name",
    "center_stem",
    "scene_judgement",
    "selected_display_labels",
    "selected_actor_ids",
    "selected_actor_classes",
    "removal_changes_action",
    "action_effect",
    "confidence",
    "notes",
    "annotation_time_s",
    "is_skipped",
    "skip_reason",
    "created_at",
    "revision",
]


def sanitize_annotator_id(value: str) -> str:
    value = "_".join(value.strip().split())
    value = "".join(ch for ch in value if ch.isalnum() or ch in "_-.")
    if not value:
        raise ValueError("annotator_id 不能为空")
    return value[:80]


@contextmanager
def _locked_file(path: Path, mode: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield f
        finally:
            f.flush()
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def annotation_paths(annotation_dir: Path, annotator_id: str) -> Dict[str, Path]:
    safe_id = sanitize_annotator_id(annotator_id)
    return {
        "jsonl": annotation_dir / f"annotations_{safe_id}.jsonl",
        "csv": annotation_dir / f"annotations_{safe_id}.csv",
    }


def load_latest_annotations(annotation_dir: Path, annotator_id: str) -> Dict[str, Dict]:
    jsonl_path = annotation_paths(annotation_dir, annotator_id)["jsonl"]
    latest: Dict[str, Dict] = {}
    if not jsonl_path.exists():
        return latest
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] 忽略损坏的标注行 {jsonl_path}:{line_no}")
                continue
            sample_id = str(record.get("sample_id", ""))
            if sample_id:
                latest[sample_id] = record
    return latest


def completed_sample_ids(annotation_dir: Path, annotator_id: str) -> Set[str]:
    return set(load_latest_annotations(annotation_dir, annotator_id).keys())


def append_annotation(annotation_dir: Path, annotator_id: str, record: Dict) -> Dict:
    paths = annotation_paths(annotation_dir, annotator_id)
    latest = load_latest_annotations(annotation_dir, annotator_id)
    previous = latest.get(str(record.get("sample_id", "")))
    revision = int(previous.get("revision", 0)) + 1 if previous else 1

    completed = dict(record)
    completed.setdefault("schema_version", 1)
    completed["annotator_id"] = sanitize_annotator_id(annotator_id)
    completed["created_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    completed["revision"] = revision

    with _locked_file(paths["jsonl"], "a") as f:
        f.write(json.dumps(completed, ensure_ascii=False) + "\n")

    csv_exists = paths["csv"].exists() and paths["csv"].stat().st_size > 0
    csv_record = dict(completed)
    for key in ["selected_display_labels", "selected_actor_ids", "selected_actor_classes"]:
        value = csv_record.get(key, [])
        if isinstance(value, list):
            csv_record[key] = "|".join(str(item) for item in value)
    with _locked_file(paths["csv"], "a") as f:
        writer = csv.DictWriter(f, fieldnames=ANNOTATION_FIELDS, extrasaction="ignore")
        if not csv_exists:
            writer.writeheader()
        writer.writerow(csv_record)

    return completed


def choose_next_index(
    samples: List[Dict],
    annotation_dir: Path,
    annotator_id: str,
    start_index: int = 0,
) -> int:
    done = completed_sample_ids(annotation_dir, annotator_id)
    for index in range(max(0, start_index), len(samples)):
        if samples[index]["sample_id"] not in done:
            return index
    for index in range(0, min(start_index, len(samples))):
        if samples[index]["sample_id"] not in done:
            return index
    return min(max(start_index, 0), max(len(samples) - 1, 0))
