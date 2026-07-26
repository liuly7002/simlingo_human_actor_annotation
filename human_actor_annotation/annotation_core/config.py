from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


DEFAULT_VIEW_ALIASES: Dict[str, List[str]] = {
    "front_left": ["rgb_front_left", "rgb_left_front"],
    "front": ["rgb_front", "rgb"],
    "front_right": ["rgb_front_right", "rgb_right_front"],
    "rear_left": ["rgb_rear_left", "rgb_left_rear", "rgb_left_back"],
    "rear": ["rgb_rear", "rgb_back"],
    "rear_right": ["rgb_rear_right", "rgb_right_rear", "rgb_right_back"],
}


@dataclass
class DatasetConfig:
    dataset_root: Path
    work_dir: Path
    manifest_path: Path
    annotation_dir: Path
    view_aliases: Dict[str, List[str]] = field(default_factory=lambda: dict(DEFAULT_VIEW_ALIASES))
    top_rgb_aliases: List[str] = field(default_factory=lambda: ["top_rgb"])
    boxes_aliases: List[str] = field(default_factory=lambda: ["boxes"])
    measurements_aliases: List[str] = field(default_factory=lambda: ["measurements"])
    image_extensions: List[str] = field(default_factory=lambda: [".png", ".jpg", ".jpeg"])
    metadata_extensions: List[str] = field(default_factory=lambda: [".json.gz", ".json"])


@dataclass
class SamplingConfig:
    # At 10 FPS, 20 history + current + 20 future frames span about 4 seconds.
    history_frames: int = 20
    future_frames: int = 20
    sample_stride: int = 40
    max_samples: int = 1000
    random_seed: int = 20260726
    require_all_six_views: bool = True
    require_top_rgb: bool = True
    require_boxes: bool = True
    max_candidates: int = 20
    skip_if_candidates_truncated: bool = True
    include_empty_candidate_samples: bool = True
    max_actor_distance_m: float = 50.0
    min_actor_forward_m: float = -8.0

    # Prefer clips in which the ego vehicle is actually moving. A small
    # stationary quota is retained because stopping at red lights, queues, or
    # obstacles is still scientifically meaningful.
    prefer_moving_samples: bool = True
    min_ego_speed_mps: float = 1.0
    min_window_moving_ratio: float = 0.50
    max_stationary_sample_ratio: float = 0.10
    keep_if_motion_unknown: bool = True

    include_classes: List[str] = field(
        default_factory=lambda: [
            "vehicle",
            "car",
            "truck",
            "bus",
            "motorcycle",
            "bicycle",
            "cyclist",
            "pedestrian",
            "walker",
            "static",
            "obstacle",
            "traffic_cone",
            "cone",
        ]
    )
    exclude_classes: List[str] = field(
        default_factory=lambda: ["ego", "ego_vehicle", "ego_car", "hero"]
    )


@dataclass
class RenderConfig:
    fps: int = 10
    mosaic_width: int = 1280
    current_mosaic_width: int = 1280
    top_preview_width: int = 720
    marker_radius_px: int = 18
    marker_anchor_radius_px: int = 5
    marker_label_gap_px: int = 10
    show_actor_class: bool = True
    show_actor_distance: bool = True
    # Pixel mapping for top_rgb when actor positions are in ego coordinates:
    # u = cx + y / meters_per_pixel, v = cy - x / meters_per_pixel.
    top_meters_per_pixel: Optional[float] = None
    top_ego_center_x_ratio: float = 0.5
    top_ego_center_y_ratio: float = 0.5
    forward_is_up: bool = True
    right_is_right: bool = True


@dataclass
class AppConfig:
    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False
    show_candidate_table: bool = True
    allow_notes: bool = True
    title: str = "关键 Actor 人工盲评标注"


@dataclass
class ProjectConfig:
    dataset: DatasetConfig
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    app: AppConfig = field(default_factory=AppConfig)


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_config(config_path: str | Path) -> ProjectConfig:
    config_path = Path(config_path).expanduser().resolve()
    base_dir = config_path.parent
    with config_path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    dataset_raw = raw.get("dataset", {})
    if "dataset_root" not in dataset_raw:
        raise ValueError("配置文件缺少 dataset.dataset_root")

    work_dir = _resolve_path(base_dir, dataset_raw.get("work_dir", "./annotation_workspace"))
    manifest_path = _resolve_path(
        base_dir,
        dataset_raw.get("manifest_path", str(work_dir / "manifest.jsonl")),
    )
    annotation_dir = _resolve_path(
        base_dir,
        dataset_raw.get("annotation_dir", str(work_dir / "annotations")),
    )

    dataset = DatasetConfig(
        dataset_root=_resolve_path(base_dir, dataset_raw["dataset_root"]),
        work_dir=work_dir,
        manifest_path=manifest_path,
        annotation_dir=annotation_dir,
        view_aliases=dataset_raw.get("view_aliases", dict(DEFAULT_VIEW_ALIASES)),
        top_rgb_aliases=dataset_raw.get("top_rgb_aliases", ["top_rgb"]),
        boxes_aliases=dataset_raw.get("boxes_aliases", ["boxes"]),
        measurements_aliases=dataset_raw.get("measurements_aliases", ["measurements"]),
        image_extensions=dataset_raw.get("image_extensions", [".png", ".jpg", ".jpeg"]),
        metadata_extensions=dataset_raw.get("metadata_extensions", [".json.gz", ".json"]),
    )

    sampling = SamplingConfig(**raw.get("sampling", {}))
    render = RenderConfig(**raw.get("render", {}))
    app = AppConfig(**raw.get("app", {}))

    dataset.work_dir.mkdir(parents=True, exist_ok=True)
    dataset.annotation_dir.mkdir(parents=True, exist_ok=True)
    dataset.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    return ProjectConfig(dataset=dataset, sampling=sampling, render=render, app=app)
