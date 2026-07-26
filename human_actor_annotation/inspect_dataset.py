from __future__ import annotations

import argparse
import json
from pathlib import Path

from annotation_core.actors import load_normalized_actors
from annotation_core.config import load_config
from annotation_core.dataset import build_route_index, discover_route_dirs, load_ego_speed_mps


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 SimLingo 数据集目录与 boxes 字段")
    parser.add_argument("--config", required=True)
    parser.add_argument("--routes", type=int, default=3, help="最多展示多少条路线")
    args = parser.parse_args()

    config = load_config(args.config)
    routes = discover_route_dirs(config)
    print(f"发现路线目录：{len(routes)}")
    for route_dir in routes[: max(1, args.routes)]:
        index = build_route_index(route_dir, config)
        print("\n" + "=" * 80)
        print(f"route: {route_dir}")
        print(f"frames: {len(index['stems'])}")
        print("view dirs:")
        for view, folder in index["view_dirs"].items():
            print(f"  {view:>12}: {folder}")
        print(f"  {'top_rgb':>12}: {index['top_dir']}")
        print(f"  {'boxes':>12}: {index['boxes_dir']}")
        print(f"  {'measurements':>12}: {index['measurements_dir']}")

        if index["stems"]:
            stem = index["stems"][len(index["stems"]) // 2]
            boxes_path = index["boxes_files"].get(stem)
            measurement_path = index["measurement_files"].get(stem)
            print(f"example frame: {stem}")
            print(f"boxes path  : {boxes_path}")
            print(f"measurement : {measurement_path}")
            print(f"ego speed   : {load_ego_speed_mps(measurement_path)} m/s")
            if boxes_path:
                try:
                    actors = load_normalized_actors(boxes_path)
                    print(f"actors      : {len(actors)}")
                    print(json.dumps(actors[:3], ensure_ascii=False, indent=2, default=str))
                except Exception as exc:
                    print(f"读取 boxes 失败：{exc}")


if __name__ == "__main__":
    main()
