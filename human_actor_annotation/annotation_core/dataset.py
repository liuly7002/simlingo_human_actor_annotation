from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .actors import load_normalized_actors, read_json_auto, select_candidates
from .config import ProjectConfig


MANIFEST_SCHEMA_VERSION = 1


VIEW_ORDER = (
    "front_left",
    "front",
    "front_right",
    "rear_left",
    "rear",
    "rear_right",
)


def _first_existing_dir(parent: Path, aliases: Sequence[str]) -> Optional[Path]:
    for alias in aliases:
        candidate = parent / alias
        if candidate.is_dir():
            return candidate
    return None


def _index_files(folder: Optional[Path], extensions: Sequence[str]) -> Dict[str, Path]:
    if folder is None or not folder.is_dir():
        return {}
    ext_set = {ext.lower() for ext in extensions}
    result: Dict[str, Path] = {}
    for path in folder.iterdir():
        if not path.is_file():
            continue
        name = path.name.lower()
        matched = False
        for ext in ext_set:
            if name.endswith(ext):
                stem = path.name[: -len(ext)] if ext else path.stem
                result[stem] = path.resolve()
                matched = True
                break
        if not matched and path.suffix.lower() in ext_set:
            result[path.stem] = path.resolve()
    return result


def _frame_sort_key(stem: str) -> Tuple[int, str]:
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        try:
            return int(digits), stem
        except ValueError:
            pass
    return 10**18, stem


def discover_route_dirs(config: ProjectConfig) -> List[Path]:
    root = config.dataset.dataset_root
    if not root.exists():
        raise FileNotFoundError(f"数据集根目录不存在：{root}")

    front_aliases = config.dataset.view_aliases.get("front", ["rgb_front", "rgb"])
    routes: List[Path] = []
    for current, dirnames, _ in os.walk(root):
        current_path = Path(current)
        if _first_existing_dir(current_path, front_aliases) is None:
            continue
        routes.append(current_path.resolve())
        # A route directory should not contain nested route directories.
        dirnames[:] = []
    routes.sort()
    return routes


def build_route_index(route_dir: Path, config: ProjectConfig) -> Dict:
    view_dirs: Dict[str, Optional[Path]] = {
        view: _first_existing_dir(route_dir, config.dataset.view_aliases.get(view, []))
        for view in VIEW_ORDER
    }
    top_dir = _first_existing_dir(route_dir, config.dataset.top_rgb_aliases)
    boxes_dir = _first_existing_dir(route_dir, config.dataset.boxes_aliases)
    measurements_dir = _first_existing_dir(route_dir, config.dataset.measurements_aliases)

    view_files = {
        view: _index_files(folder, config.dataset.image_extensions)
        for view, folder in view_dirs.items()
    }
    top_files = _index_files(top_dir, config.dataset.image_extensions)
    boxes_files = _index_files(boxes_dir, config.dataset.metadata_extensions)
    measurement_files = _index_files(measurements_dir, config.dataset.metadata_extensions)

    front_stems = set(view_files["front"].keys())
    if config.sampling.require_all_six_views:
        for view in VIEW_ORDER:
            front_stems &= set(view_files[view].keys())
    if config.sampling.require_top_rgb:
        front_stems &= set(top_files.keys())
    if config.sampling.require_boxes:
        front_stems &= set(boxes_files.keys())

    stems = sorted(front_stems, key=_frame_sort_key)
    return {
        "route_dir": route_dir,
        "view_dirs": view_dirs,
        "top_dir": top_dir,
        "boxes_dir": boxes_dir,
        "measurements_dir": measurements_dir,
        "view_files": view_files,
        "top_files": top_files,
        "boxes_files": boxes_files,
        "measurement_files": measurement_files,
        "stems": stems,
    }


def route_name(route_dir: Path, dataset_root: Path) -> str:
    try:
        relative = route_dir.relative_to(dataset_root)
        name = str(relative).replace(os.sep, "__")
    except ValueError:
        name = route_dir.name
    return name or route_dir.name


def _relative_or_absolute(path: Optional[Path], base: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def resolve_manifest_path(path_value: Optional[str], dataset_root: Path) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (dataset_root / path).resolve()



def _safe_float(value) -> Optional[float]:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _vector_speed(value) -> Optional[float]:
    if isinstance(value, dict):
        components = [_safe_float(value.get(key)) for key in ("x", "y", "z")]
        finite = [component for component in components if component is not None]
        if len(finite) >= 2:
            return float(math.sqrt(sum(component * component for component in finite)))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        components = [_safe_float(component) for component in value[:3]]
        finite = [component for component in components if component is not None]
        if len(finite) >= 2:
            return float(math.sqrt(sum(component * component for component in finite)))
    return None


def _extract_ego_speed_mps(data) -> Optional[float]:
    """Read actual ego speed from common SimLingo/CARLA measurement layouts."""
    if not isinstance(data, dict):
        return None

    scalar_keys = (
        "speed",
        "speed_mps",
        "ego_speed",
        "ego_speed_mps",
        "vehicle_speed",
        "speedometer",
    )
    for key in scalar_keys:
        if key in data:
            value = _safe_float(data.get(key))
            if value is not None and value >= 0.0:
                return value

    for key in ("velocity", "ego_velocity", "linear_velocity"):
        if key in data:
            value = _vector_speed(data.get(key))
            if value is not None:
                return value

    # Prefer semantically named nested containers instead of recursively
    # searching every dictionary, which could accidentally read target speed.
    for key in ("ego", "vehicle", "state", "measurement", "measurements"):
        nested = data.get(key)
        if isinstance(nested, dict):
            value = _extract_ego_speed_mps(nested)
            if value is not None:
                return value
    return None


def load_ego_speed_mps(path: Optional[Path]) -> Optional[float]:
    if path is None or not path.exists():
        return None
    try:
        return _extract_ego_speed_mps(read_json_auto(path))
    except Exception:
        return None


def _motion_summary(
    window_stems: Sequence[str],
    center_stem: str,
    measurement_files: Dict[str, Path],
    speed_cache: Dict[str, Optional[float]],
    min_speed_mps: float,
) -> Dict:
    def speed_for(stem: str) -> Optional[float]:
        if stem not in speed_cache:
            speed_cache[stem] = load_ego_speed_mps(measurement_files.get(stem))
        return speed_cache[stem]

    speeds = [speed_for(stem) for stem in window_stems]
    valid = [float(speed) for speed in speeds if speed is not None]
    center_speed = speed_for(center_stem)
    if not valid:
        return {
            "status": "unknown",
            "center_speed_mps": center_speed,
            "mean_speed_mps": None,
            "moving_frame_ratio": None,
            "valid_speed_frames": 0,
            "total_frames": len(window_stems),
        }

    moving_ratio = sum(speed >= min_speed_mps for speed in valid) / len(valid)
    return {
        "status": "measured",
        "center_speed_mps": center_speed,
        "mean_speed_mps": sum(valid) / len(valid),
        "moving_frame_ratio": moving_ratio,
        "valid_speed_frames": len(valid),
        "total_frames": len(window_stems),
    }

def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp.replace(path)


def load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 第 {line_no} 行格式错误：{path}") from exc
    return records


def prepare_manifest(config: ProjectConfig) -> Dict[str, int]:
    routes = discover_route_dirs(config)
    dataset_root = config.dataset.dataset_root
    sampling = config.sampling

    records: List[Dict] = []
    stationary_limit = max(
        0,
        int(math.floor(float(sampling.max_samples) * float(sampling.max_stationary_sample_ratio))),
    )
    stationary_written = 0
    stats = {
        "routes_found": len(routes),
        "routes_with_samples": 0,
        "frames_seen": 0,
        "samples_written": 0,
        "moving_samples_written": 0,
        "stationary_samples_written": 0,
        "motion_unknown_samples_written": 0,
        "skipped_stationary": 0,
        "skipped_motion_unknown": 0,
        "skipped_short_window": 0,
        "skipped_no_boxes": 0,
        "skipped_no_candidates": 0,
        "empty_candidate_samples": 0,
        "skipped_truncated_candidates": 0,
        "actor_truncated_total": 0,
    }

    for route_dir in routes:
        route_index = build_route_index(route_dir, config)
        stems: List[str] = route_index["stems"]
        stats["frames_seen"] += len(stems)
        if not stems:
            continue

        speed_cache: Dict[str, Optional[float]] = {}
        route_sample_count = 0
        start = sampling.history_frames
        stop = max(start, len(stems) - sampling.future_frames)
        for center_idx in range(start, stop, max(1, sampling.sample_stride)):
            if len(records) >= sampling.max_samples:
                break

            window_stems = stems[
                center_idx - sampling.history_frames : center_idx + sampling.future_frames + 1
            ]
            expected = sampling.history_frames + sampling.future_frames + 1
            if len(window_stems) != expected:
                stats["skipped_short_window"] += 1
                continue

            center_stem = stems[center_idx]
            motion = _motion_summary(
                window_stems=window_stems,
                center_stem=center_stem,
                measurement_files=route_index["measurement_files"],
                speed_cache=speed_cache,
                min_speed_mps=float(sampling.min_ego_speed_mps),
            )
            motion_status = str(motion.get("status", "unknown"))
            center_speed = motion.get("center_speed_mps")
            moving_ratio = motion.get("moving_frame_ratio")
            is_moving = (
                motion_status == "measured"
                and (
                    (center_speed is not None and float(center_speed) >= float(sampling.min_ego_speed_mps))
                    or (
                        moving_ratio is not None
                        and float(moving_ratio) >= float(sampling.min_window_moving_ratio)
                    )
                )
            )
            if sampling.prefer_moving_samples:
                if motion_status == "unknown":
                    if not sampling.keep_if_motion_unknown:
                        stats["skipped_motion_unknown"] += 1
                        continue
                elif not is_moving:
                    if stationary_written >= stationary_limit:
                        stats["skipped_stationary"] += 1
                        continue

            boxes_path = route_index["boxes_files"].get(center_stem)
            if boxes_path is None:
                stats["skipped_no_boxes"] += 1
                continue

            sample_id = f"{route_name(route_dir, dataset_root)}__{center_stem}"
            try:
                actors = load_normalized_actors(boxes_path)
            except Exception as exc:
                stats["skipped_no_boxes"] += 1
                print(f"[WARN] 无法读取 boxes，跳过 {sample_id}: {exc}")
                continue

            candidates, candidate_stats = select_candidates(
                actors=actors,
                include_classes=sampling.include_classes,
                exclude_classes=sampling.exclude_classes,
                max_distance_m=sampling.max_actor_distance_m,
                min_forward_m=sampling.min_actor_forward_m,
                max_candidates=sampling.max_candidates,
                sample_id=sample_id,
                random_seed=sampling.random_seed,
            )
            truncated_count = int(candidate_stats.get("truncated", 0))
            stats["actor_truncated_total"] += truncated_count
            if truncated_count > 0 and sampling.skip_if_candidates_truncated:
                stats["skipped_truncated_candidates"] += 1
                continue
            if not candidates:
                if not sampling.include_empty_candidate_samples:
                    stats["skipped_no_candidates"] += 1
                    continue
                stats["empty_candidate_samples"] += 1

            # Remove original raw dictionaries/matrices from the manifest.
            # Human annotation only needs normalized, method-independent fields.
            candidates = [
                {
                    key: actor.get(key)
                    for key in [
                        "actor_id", "class", "raw_class", "x_m", "y_m",
                        "distance_m", "speed_mps", "role_name", "extent",
                        "display_label",
                    ]
                }
                for actor in candidates
            ]

            frame_records: List[Dict] = []
            complete_window = True
            for stem in window_stems:
                views = {}
                for view in VIEW_ORDER:
                    image_path = route_index["view_files"][view].get(stem)
                    if image_path is None and sampling.require_all_six_views:
                        complete_window = False
                        break
                    views[view] = _relative_or_absolute(image_path, dataset_root)
                if not complete_window:
                    break
                frame_records.append(
                    {
                        "stem": stem,
                        "views": views,
                        "top_rgb": _relative_or_absolute(
                            route_index["top_files"].get(stem), dataset_root
                        ),
                    }
                )
            if not complete_window:
                stats["skipped_short_window"] += 1
                continue

            center_measurement = route_index["measurement_files"].get(center_stem)
            motion_class = "moving" if is_moving else (
                "unknown" if motion_status == "unknown" else "stationary"
            )
            record = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "sample_id": sample_id,
                "route_name": route_name(route_dir, dataset_root),
                "route_dir": _relative_or_absolute(route_dir, dataset_root),
                "center_stem": center_stem,
                "center_index": center_idx,
                "history_frames": sampling.history_frames,
                "future_frames": sampling.future_frames,
                "frames": frame_records,
                "center_frame_offset": sampling.history_frames,
                "boxes_path": _relative_or_absolute(boxes_path, dataset_root),
                "measurement_path": _relative_or_absolute(center_measurement, dataset_root),
                "motion_class": motion_class,
                "motion_summary": motion,
                "candidates": candidates,
                "candidate_filter_stats": candidate_stats,
                "blind_annotation": True,
            }
            records.append(record)
            route_sample_count += 1
            if motion_class == "moving":
                stats["moving_samples_written"] += 1
            elif motion_class == "stationary":
                stationary_written += 1
                stats["stationary_samples_written"] += 1
            else:
                stats["motion_unknown_samples_written"] += 1

        if route_sample_count:
            stats["routes_with_samples"] += 1
        if len(records) >= sampling.max_samples:
            break

    # Stable global shuffle prevents route-order effects while preserving reproducibility.
    import random

    rng = random.Random(sampling.random_seed)
    rng.shuffle(records)
    write_jsonl(config.dataset.manifest_path, records)
    stats["samples_written"] = len(records)

    summary_path = config.dataset.work_dir / "prepare_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "stats": stats,
                "dataset_root": str(dataset_root),
                "manifest_path": str(config.dataset.manifest_path),
                "sampling": asdict(sampling),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return stats
