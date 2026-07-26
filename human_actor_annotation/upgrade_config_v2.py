from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将第一版 config.yaml 复制并升级为第二版 4 秒/行驶优先配置"
    )
    parser.add_argument("--config", required=True, help="现有 config.yaml")
    parser.add_argument(
        "--output",
        default="",
        help="输出路径；默认在原文件旁生成 config_v2.yaml",
    )
    args = parser.parse_args()

    source = Path(args.config).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"配置文件不存在：{source}")
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else source.with_name(f"{source.stem}_v2{source.suffix}")
    )

    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    dataset = raw.setdefault("dataset", {})
    old_work_dir = Path(str(dataset.get("work_dir", "./annotation_workspace")))
    new_work_dir = old_work_dir.with_name(old_work_dir.name + "_v2")
    dataset["work_dir"] = str(new_work_dir)
    dataset["manifest_path"] = str(new_work_dir / "manifest.jsonl")
    dataset["annotation_dir"] = str(new_work_dir / "annotations")

    sampling = raw.setdefault("sampling", {})
    sampling.update(
        {
            "history_frames": 20,
            "future_frames": 20,
            "sample_stride": 40,
            "prefer_moving_samples": True,
            "min_ego_speed_mps": 1.0,
            "min_window_moving_ratio": 0.50,
            "max_stationary_sample_ratio": 0.10,
            "keep_if_motion_unknown": True,
        }
    )
    render = raw.setdefault("render", {})
    render.setdefault("marker_radius_px", 18)
    render.setdefault("marker_anchor_radius_px", 5)
    render.setdefault("marker_label_gap_px", 10)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"已生成第二版配置：{output}")
    print(f"第二版工作目录：{new_work_dir}（不会混入第一版标注结果）")
    print("下一步：")
    print(f"  python prepare_samples.py --config {output}")
    print(f"  python annotation_app.py --config {output} --annotator-id P01")


if __name__ == "__main__":
    main()
