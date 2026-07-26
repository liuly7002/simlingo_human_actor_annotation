from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np


VIEWS = [
    "front_left",
    "front",
    "front_right",
    "rear_left",
    "rear",
    "rear_right",
]


def make_image(view: str, frame: int, width: int = 640, height: int = 360) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    base = 35 + (VIEWS.index(view) * 22) % 160
    image[:] = (base, 70, 150)
    cv2.putText(image, view, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (245, 245, 245), 2)
    cv2.putText(image, f"frame {frame:04d}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 245, 245), 2)
    # Simple moving shapes to make the context video visibly temporal.
    cv2.rectangle(image, (80 + frame * 3 % 400, 210), (160 + frame * 3 % 400, 280), (30, 220, 220), -1)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="demo_dataset/data/Town12/demo_route")
    args = parser.parse_args()

    route = Path(args.output).resolve()
    for view in VIEWS:
        (route / f"rgb_{view}").mkdir(parents=True, exist_ok=True)
    for folder in ["top_rgb", "boxes", "measurements"]:
        (route / folder).mkdir(parents=True, exist_ok=True)

    with (route / "surround_camera_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "top_rgb": {
                    "position": [0.0, 0.0, 40.0],
                    "width": 600,
                    "height": 600,
                    "fov": 90.0,
                }
            },
            f,
            indent=2,
        )

    for frame in range(50):
        stem = f"{frame:04d}"
        for view in VIEWS:
            cv2.imwrite(str(route / f"rgb_{view}" / f"{stem}.jpg"), make_image(view, frame))

        top = np.full((600, 600, 3), 45, dtype=np.uint8)
        cv2.line(top, (300, 600), (300, 0), (140, 140, 140), 5)
        cv2.line(top, (230, 600), (230, 0), (90, 90, 90), 2)
        cv2.line(top, (370, 600), (370, 0), (90, 90, 90), 2)
        cv2.putText(top, f"top {stem}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
        cv2.imwrite(str(route / "top_rgb" / f"{stem}.jpg"), top)

        boxes = [
            {
                "id": 999,
                "class": "ego_car",
                "position": [0.0, 0.0, 0.0],
                "role_name": "hero",
            },
            {
                "id": 101,
                "class": "vehicle",
                "position": [14.0 + 0.04 * frame, -1.2, 0.0],
                "speed": 2.5,
            },
            {
                "id": 202,
                "class": "pedestrian",
                "position": [9.0, 4.0 - 0.08 * frame, 0.0],
                "speed": 1.0,
            },
            {
                "id": 303,
                "class": "vehicle",
                "position": [-3.0, 3.0, 0.0],
                "speed": 4.0,
            },
            # Dense cluster used to verify label collision avoidance.
            {
                "id": 404,
                "class": "vehicle",
                "position": [14.2, -1.0, 0.0],
                "speed": 2.2,
            },
            {
                "id": 505,
                "class": "vehicle",
                "position": [13.8, -1.4, 0.0],
                "speed": 2.0,
            },
            # Slightly outside the top-camera footprint; its label must stay inside.
            {
                "id": 606,
                "class": "vehicle",
                "position": [41.0, 0.0, 0.0],
                "speed": 3.0,
            },
        ]
        with gzip.open(route / "boxes" / f"{stem}.json.gz", "wt", encoding="utf-8") as f:
            json.dump(boxes, f)
        with gzip.open(route / "measurements" / f"{stem}.json.gz", "wt", encoding="utf-8") as f:
            json.dump({"speed": 5.0, "frame": frame}, f)

    print(route)


if __name__ == "__main__":
    main()
