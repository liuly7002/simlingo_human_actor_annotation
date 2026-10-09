from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import ProjectConfig
from .dataset import VIEW_ORDER, resolve_manifest_path


VIEW_TITLES = {
    "front_left": "Front-left",
    "front": "Front",
    "front_right": "Front-right",
    "rear_left": "Rear-left",
    "rear": "Rear",
    "rear_right": "Rear-right",
}


def _safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value)[:180]


def _read_image(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None or not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return image


def _letterbox(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    scale = min(target_w / w, target_h / h)
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y = (target_h - resized.shape[0]) // 2
    x = (target_w - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def make_surround_mosaic(
    frame: Dict,
    dataset_root: Path,
    target_width: int,
) -> np.ndarray:
    paths: Dict[str, Optional[Path]] = {
        view: resolve_manifest_path(frame.get("views", {}).get(view), dataset_root)
        for view in VIEW_ORDER
    }
    loaded = {view: _read_image(path) for view, path in paths.items()}
    valid = [image for image in loaded.values() if image is not None]
    if not valid:
        return np.zeros((720, target_width, 3), dtype=np.uint8)

    source_h, source_w = valid[0].shape[:2]
    cell_w = max(240, target_width // 3)
    cell_h = max(160, int(cell_w * source_h / max(source_w, 1)))
    label_h = 34

    rows: List[np.ndarray] = []
    for row_views in [VIEW_ORDER[:3], VIEW_ORDER[3:]]:
        cells = []
        for view in row_views:
            image = loaded.get(view)
            if image is None:
                image = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                cv2.putText(
                    image,
                    "MISSING",
                    (20, cell_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (220, 220, 220),
                    2,
                    cv2.LINE_AA,
                )
            else:
                image = _letterbox(image, cell_w, cell_h)
            header = np.full((label_h, cell_w, 3), 28, dtype=np.uint8)
            cv2.putText(
                header,
                VIEW_TITLES[view],
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            cells.append(np.vstack([header, image]))
        rows.append(np.hstack(cells))
    mosaic = np.vstack(rows)
    if mosaic.shape[1] != target_width:
        new_h = int(mosaic.shape[0] * target_width / mosaic.shape[1])
        mosaic = cv2.resize(mosaic, (target_width, new_h))
    return mosaic


def _load_top_camera_metadata(route_dir: Path) -> Dict:
    metadata_path = route_dir / "surround_camera_config.json"
    if not metadata_path.exists():
        return {}
    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _derive_meters_per_pixel(
    top_image: np.ndarray,
    route_dir: Path,
    config: ProjectConfig,
) -> float:
    if config.render.top_meters_per_pixel is not None:
        return float(config.render.top_meters_per_pixel)

    metadata = _load_top_camera_metadata(route_dir)
    top = metadata.get("top_rgb", {}) if isinstance(metadata, dict) else {}
    try:
        z = float(top.get("position", [0.0, 0.0, 40.0])[2])
        fov = float(top.get("fov", 90.0))
        configured_width = float(top.get("width", top_image.shape[1]))
        ground_width = 2.0 * z * math.tan(math.radians(fov) / 2.0)
        native_mpp = ground_width / max(configured_width, 1.0)
        # Account for an image whose saved width differs from metadata.
        return native_mpp * configured_width / max(float(top_image.shape[1]), 1.0)
    except Exception:
        # Typical 40 m / 90 degree top camera over a 512 px image.
        return 80.0 / max(float(top_image.shape[1]), 1.0)


def _actor_pixel(
    actor: Dict,
    width: int,
    height: int,
    meters_per_pixel: float,
    config: ProjectConfig,
) -> Optional[Tuple[int, int, bool]]:
    """Return a safe anchor pixel and whether the true center was clipped.

    A candidate slightly outside the top image is anchored to the closest image
    edge instead of drawing a half-visible label outside the canvas. Candidates
    far beyond the top-camera coverage are omitted from the overlay but remain
    available in the candidate table and six-view video.
    """
    x = actor.get("x_m")
    y = actor.get("y_m")
    if x is None or y is None:
        return None

    cx = config.render.top_ego_center_x_ratio * width
    cy = config.render.top_ego_center_y_ratio * height
    right_sign = 1.0 if config.render.right_is_right else -1.0
    forward_sign = -1.0 if config.render.forward_is_up else 1.0
    raw_u = cx + right_sign * float(y) / meters_per_pixel
    raw_v = cy + forward_sign * float(x) / meters_per_pixel

    outside_tolerance = max(40.0, 2.5 * float(config.render.marker_radius_px))
    if not (
        -outside_tolerance <= raw_u <= width - 1 + outside_tolerance
        and -outside_tolerance <= raw_v <= height - 1 + outside_tolerance
    ):
        return None

    anchor_margin = max(3, int(config.render.marker_anchor_radius_px) + 2)
    u = int(round(np.clip(raw_u, anchor_margin, max(anchor_margin, width - 1 - anchor_margin))))
    v = int(round(np.clip(raw_v, anchor_margin, max(anchor_margin, height - 1 - anchor_margin))))
    clipped = not (0.0 <= raw_u <= width - 1 and 0.0 <= raw_v <= height - 1)
    return u, v, clipped


def _label_overlap_penalty(
    center: Tuple[int, int],
    radius: int,
    occupied: List[Tuple[Tuple[int, int], int]],
) -> float:
    penalty = 0.0
    for other_center, other_radius in occupied:
        distance = math.hypot(center[0] - other_center[0], center[1] - other_center[1])
        required = radius + other_radius + 4
        if distance < required:
            penalty += (required - distance) ** 2
    return penalty


def _choose_label_center(
    anchor: Tuple[int, int],
    width: int,
    height: int,
    radius: int,
    gap: int,
    occupied: List[Tuple[Tuple[int, int], int]],
) -> Tuple[int, int]:
    margin = radius + 3
    base_distance = 2 * radius + max(4, gap)
    # Put labels around the physical anchor. Several rings allow dense clusters
    # to spread without overlapping while leader lines preserve correspondence.
    angles_deg = (-90, -45, 0, 45, 90, 135, 180, 225, 270, 315)
    candidates: List[Tuple[int, int, float]] = []
    for ring in range(1, 6):
        distance = base_distance * ring
        for angle_deg in angles_deg:
            angle = math.radians(angle_deg)
            raw_x = anchor[0] + distance * math.cos(angle)
            raw_y = anchor[1] + distance * math.sin(angle)
            x = int(round(np.clip(raw_x, margin, max(margin, width - 1 - margin))))
            y = int(round(np.clip(raw_y, margin, max(margin, height - 1 - margin))))
            boundary_shift = math.hypot(x - raw_x, y - raw_y)
            overlap = _label_overlap_penalty((x, y), radius, occupied)
            score = overlap * 1000.0 + boundary_shift * 20.0 + distance
            candidates.append((x, y, score))

    # A compact image with many actors may still be crowded. Choose the least
    # conflicting valid position rather than allowing labels to leave the image.
    x, y, _ = min(candidates, key=lambda item: item[2])
    return x, y


def _layout_candidate_labels(
    sample: Dict,
    width: int,
    height: int,
    meters_per_pixel: float,
    config: ProjectConfig,
) -> List[Dict]:
    radius = max(10, int(config.render.marker_radius_px))
    gap = int(config.render.marker_label_gap_px)
    ego_center = (
        int(round(config.render.top_ego_center_x_ratio * width)),
        int(round(config.render.top_ego_center_y_ratio * height)),
    )
    occupied: List[Tuple[Tuple[int, int], int]] = [(ego_center, 16)]
    placements: List[Dict] = []
    for actor in sample.get("candidates", []):
        pixel = _actor_pixel(actor, width, height, meters_per_pixel, config)
        if pixel is None:
            continue
        anchor_u, anchor_v, clipped = pixel
        label_center = _choose_label_center(
            (anchor_u, anchor_v), width, height, radius, gap, occupied
        )
        occupied.append((label_center, radius))
        placements.append(
            {
                "actor": actor,
                "anchor": (anchor_u, anchor_v),
                "label_center": label_center,
                "clipped": clipped,
                "radius": radius,
            }
        )
    return placements


def annotate_top_rgb(sample: Dict, config: ProjectConfig) -> np.ndarray:
    dataset_root = config.dataset.dataset_root
    center_offset = int(sample["center_frame_offset"])
    frame = sample["frames"][center_offset]
    top_path = resolve_manifest_path(frame.get("top_rgb"), dataset_root)
    top = _read_image(top_path)
    if top is None:
        top = np.zeros((720, 720, 3), dtype=np.uint8)
        cv2.putText(
            top,
            "TOP_RGB MISSING",
            (80, top.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )

    route_dir = resolve_manifest_path(sample.get("route_dir"), dataset_root)
    route_dir = route_dir if route_dir is not None else dataset_root
    mpp = _derive_meters_per_pixel(top, route_dir, config)

    marker_color = (30, 215, 255)
    text_color = (20, 20, 20)
    leader_color = (240, 240, 240)
    anchor_color = (20, 20, 20)
    anchor_radius = max(3, int(config.render.marker_anchor_radius_px))
    placements = _layout_candidate_labels(
        sample, top.shape[1], top.shape[0], mpp, config
    )

    # Draw leader lines first so bubbles and anchors remain clean.
    for item in placements:
        cv2.line(
            top,
            item["anchor"],
            item["label_center"],
            leader_color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    for item in placements:
        anchor = item["anchor"]
        label_center = item["label_center"]
        radius = int(item["radius"])
        actor = item["actor"]
        label = str(actor.get("display_label", "A?"))

        if item["clipped"]:
            cv2.drawMarker(
                top,
                anchor,
                marker_color,
                markerType=cv2.MARKER_DIAMOND,
                markerSize=anchor_radius * 3,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
        else:
            cv2.circle(top, anchor, anchor_radius + 2, (245, 245, 245), -1, cv2.LINE_AA)
            cv2.circle(top, anchor, anchor_radius, anchor_color, -1, cv2.LINE_AA)

        cv2.circle(top, label_center, radius, marker_color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(top, label_center, radius, (10, 10, 10), thickness=2, lineType=cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        cv2.putText(
            top,
            label,
            (label_center[0] - tw // 2, label_center[1] + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            text_color,
            2,
            cv2.LINE_AA,
        )

    # Mark ego position without adding an actor candidate.
    ego_u = int(round(config.render.top_ego_center_x_ratio * top.shape[1]))
    ego_v = int(round(config.render.top_ego_center_y_ratio * top.shape[0]))
    cv2.drawMarker(
        top,
        (ego_u, ego_v),
        (255, 255, 255),
        markerType=cv2.MARKER_TRIANGLE_UP,
        markerSize=24,
        thickness=2,
        line_type=cv2.LINE_AA,
    )

    target_w = int(config.render.top_preview_width)
    if target_w > 0 and top.shape[1] != target_w:
        target_h = max(1, int(top.shape[0] * target_w / top.shape[1]))
        top = cv2.resize(top, (target_w, target_h))
    return top


def candidate_table(sample: Dict) -> List[List[str]]:
    rows: List[List[str]] = []
    for actor in sample.get("candidates", []):
        x = actor.get("x_m")
        y = actor.get("y_m")
        distance = actor.get("distance_m")
        speed = actor.get("speed_mps")
        rows.append(
            [
                str(actor.get("display_label", "")),
                str(actor.get("class", "unknown")),
                "" if x is None else f"{float(x):.1f}",
                "" if y is None else f"{float(y):.1f}",
                "" if distance is None else f"{float(distance):.1f}",
                "" if speed is None else f"{float(speed):.1f}",
            ]
        )
    return rows


def _find_ffmpeg() -> Optional[str]:
    """Return an FFmpeg executable from PATH or imageio-ffmpeg."""
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg  # type: ignore

        candidate = imageio_ffmpeg.get_ffmpeg_exe()
        return candidate if candidate and Path(candidate).exists() else None
    except Exception:
        return None


def _write_browser_video(
    mosaics: List[np.ndarray],
    video_path: Path,
    fps: int,
) -> Tuple[bool, str]:
    """Write an H.264/yuv420p MP4 that Chrome and Gradio can decode.

    OpenCV's ``mp4v`` output is used only as an intermediate file.  MPEG-4
    Part 2 video inside MP4 is not reliably supported by modern browsers.
    """
    if not mosaics:
        return False, "没有可写入的视频帧"

    frame_h, frame_w = mosaics[0].shape[:2]
    # H.264 with yuv420p requires even dimensions.
    frame_w -= frame_w % 2
    frame_h -= frame_h % 2
    if frame_w <= 0 or frame_h <= 0:
        return False, "视频帧尺寸无效"

    raw_path = video_path.with_name(video_path.stem + "_opencv_raw.mp4")
    raw_path.unlink(missing_ok=True)
    video_path.unlink(missing_ok=True)

    writer = cv2.VideoWriter(
        str(raw_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1, int(fps)),
        (frame_w, frame_h),
    )
    if not writer.isOpened():
        writer.release()
        return False, "OpenCV VideoWriter 无法启动"

    try:
        for mosaic in mosaics:
            if mosaic.shape[:2] != (frame_h, frame_w):
                mosaic = cv2.resize(mosaic, (frame_w, frame_h))
            writer.write(mosaic)
    finally:
        writer.release()

    if not raw_path.exists() or raw_path.stat().st_size < 1024:
        raw_path.unlink(missing_ok=True)
        return False, "OpenCV 中间视频为空"

    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        # Keep the raw file for diagnosis, but do not advertise it as a
        # browser-compatible result.
        return False, (
            "未找到 FFmpeg。请安装 ffmpeg 或 imageio-ffmpeg，以生成浏览器可播放的 H.264 MP4。"
        )

    commands = [
        [
            ffmpeg, "-y", "-loglevel", "error", "-i", str(raw_path),
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_path),
        ],
        [
            ffmpeg, "-y", "-loglevel", "error", "-i", str(raw_path),
            "-an", "-c:v", "h264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(video_path),
        ],
    ]
    last_error = ""
    for command in commands:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode == 0 and video_path.exists() and video_path.stat().st_size >= 1024:
            raw_path.unlink(missing_ok=True)
            return True, "H.264 MP4 已生成"
        last_error = completed.stderr.strip()[-1000:]
        video_path.unlink(missing_ok=True)

    return False, f"FFmpeg H.264 转码失败：{last_error}"


def _render_signature(sample: Dict, config: ProjectConfig) -> str:
    payload = {
        "version": 3,
        "sample_id": sample.get("sample_id"),
        "center": sample.get("center_stem"),
        "frames": [frame.get("stem") for frame in sample.get("frames", [])],
        "candidates": [
            [
                actor.get("display_label"),
                actor.get("actor_id"),
                actor.get("x_m"),
                actor.get("y_m"),
            ]
            for actor in sample.get("candidates", [])
        ],
        "render": {
            "fps": config.render.fps,
            "mosaic_width": config.render.mosaic_width,
            "current_mosaic_width": config.render.current_mosaic_width,
            "top_preview_width": config.render.top_preview_width,
            "marker_radius_px": config.render.marker_radius_px,
            "marker_anchor_radius_px": config.render.marker_anchor_radius_px,
            "marker_label_gap_px": config.render.marker_label_gap_px,
            "top_meters_per_pixel": config.render.top_meters_per_pixel,
        },
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:14]


def render_sample_assets(sample: Dict, config: ProjectConfig) -> Dict[str, str | List[List[str]]]:
    cache_dir = config.dataset.work_dir / "cache" / _safe_name(sample["sample_id"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = _render_signature(sample, config)
    current_path = cache_dir / f"current_surround_{signature}.jpg"
    top_path = cache_dir / f"top_candidates_{signature}.jpg"
    video_path = cache_dir / f"surround_context_{signature}.mp4"
    video_marker = cache_dir / f"surround_context_{signature}.ok"

    center = sample["frames"][int(sample["center_frame_offset"])]
    if not current_path.exists():
        current = make_surround_mosaic(
            center,
            config.dataset.dataset_root,
            int(config.render.current_mosaic_width),
        )
        cv2.imwrite(str(current_path), current)

    if not top_path.exists():
        top = annotate_top_rgb(sample, config)
        cv2.imwrite(str(top_path), top)

    needs_video = (
        not video_path.exists()
        or video_path.stat().st_size < 1024
        or not video_marker.exists()
    )
    if needs_video:
        video_marker.unlink(missing_ok=True)
        mosaics = [
            make_surround_mosaic(
                frame,
                config.dataset.dataset_root,
                int(config.render.mosaic_width),
            )
            for frame in sample["frames"]
        ]
        success, message = _write_browser_video(
            mosaics,
            video_path,
            max(1, int(config.render.fps)),
        )
        if success:
            video_marker.write_text(message + "\n", encoding="utf-8")
        else:
            print(f"[Video] {sample['sample_id']}: {message}")

    return {
        "video": str(video_path) if video_path.exists() and video_marker.exists() else "",
        "current": str(current_path),
        "top": str(top_path),
        "table": candidate_table(sample),
    }

