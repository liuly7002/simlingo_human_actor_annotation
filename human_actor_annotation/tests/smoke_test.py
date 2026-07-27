from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEMO_ROOT = ROOT / "demo_dataset"
DEMO_WORKSPACE = ROOT / "demo_workspace"
CONFIG = ROOT / "config.demo.yaml"


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    shutil.rmtree(DEMO_ROOT, ignore_errors=True)
    shutil.rmtree(DEMO_WORKSPACE, ignore_errors=True)
    CONFIG.unlink(missing_ok=True)

    run("tests/make_demo_dataset.py", "--output", str(DEMO_ROOT / "data/Town12/demo_route"))
    CONFIG.write_text(
        f"""dataset:
  dataset_root: {DEMO_ROOT / 'data'}
  work_dir: {DEMO_WORKSPACE}
  manifest_path: {DEMO_WORKSPACE / 'manifest.jsonl'}
  annotation_dir: {DEMO_WORKSPACE / 'annotations'}
sampling:
  history_frames: 40
  future_frames: 0
  sample_stride: 40
  max_samples: 1
  max_candidates: 12
  prefer_moving_samples: true
  min_ego_speed_mps: 1.0
  min_window_moving_ratio: 0.5
  max_stationary_sample_ratio: 0.1
render:
  fps: 10
  mosaic_width: 900
  current_mosaic_width: 900
  top_preview_width: 500
app:
  host: 127.0.0.1
  port: 7860
""",
        encoding="utf-8",
    )
    run("inspect_dataset.py", "--config", str(CONFIG), "--routes", "1")
    run("prepare_samples.py", "--config", str(CONFIG))

    from annotation_core.config import load_config
    from annotation_core.dataset import load_jsonl
    from annotation_core.metrics import (
        build_consensus,
        compare_predictions,
        krippendorff_alpha_binary,
        pairwise_agreement,
    )
    from annotation_core.render import render_sample_assets
    from annotation_core.storage import append_annotation

    cfg = load_config(CONFIG)
    samples = load_jsonl(cfg.dataset.manifest_path)
    assert samples, "manifest should contain samples"
    assert len(samples[0]["frames"]) == 41, "4-second history clip should contain 41 frames at 10 FPS"
    assert samples[0]["history_frames"] == 40
    assert samples[0]["future_frames"] == 0
    assert samples[0]["center_frame_offset"] == 40
    assert samples[0]["center_frame_offset"] == len(samples[0]["frames"]) - 1
    assert samples[0]["frames"][-1]["stem"] == samples[0]["center_stem"]
    assert samples[0].get("motion_class") == "moving"

    from annotation_core.render import _layout_candidate_labels

    placements = _layout_candidate_labels(samples[0], 600, 600, 80.0 / 600.0, cfg)
    radius = int(cfg.render.marker_radius_px)
    for item in placements:
        x, y = item["label_center"]
        assert radius <= x <= 599 - radius
        assert radius <= y <= 599 - radius
    for i, item in enumerate(placements):
        for other in placements[i + 1:]:
            x1, y1 = item["label_center"]
            x2, y2 = other["label_center"]
            assert ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 >= 2 * radius

    assets = render_sample_assets(samples[0], cfg)
    assert Path(assets["current"]).exists()
    assert Path(assets["top"]).exists()
    assert Path(assets["video"]).exists()

    candidate = samples[0]["candidates"][0]
    base = {
        "sample_id": samples[0]["sample_id"],
        "route_name": samples[0]["route_name"],
        "center_stem": samples[0]["center_stem"],
        "attention_status": "actors",
        "selected_display_labels": [candidate["display_label"]],
        "selected_actor_ids": [candidate["actor_id"]],
        "selected_actor_classes": [candidate["class"]],
        "confidence": 4,
        "notes": "",
        "annotation_time_s": 3.2,
        "is_skipped": False,
        "skip_reason": "",
    }
    r1 = append_annotation(cfg.dataset.annotation_dir, "P01", base)
    r2 = append_annotation(cfg.dataset.annotation_dir, "P02", base)

    consensus = build_consensus([r1, r2])
    actor_id = str(candidate["actor_id"])
    assert consensus[0]["consensus_actor_ids"] == [actor_id]
    assert consensus[0]["consensus_primary_actor_id"] == actor_id

    agreement = pairwise_agreement([r1, r2])
    assert agreement["pairwise_exact_set_agreement"] == 1.0
    assert agreement["pairwise_mean_jaccard"] == 1.0
    alpha = krippendorff_alpha_binary([r1, r2], samples)
    assert alpha != float("inf")

    comparison = compare_predictions(
        consensus,
        [
            {
                "sample_id": samples[0]["sample_id"],
                "selected_actor_id": actor_id,
                "candidate_actor_ids": [actor_id],
            }
        ],
        candidate_k=3,
    )
    assert comparison["human_actor_consensus_samples"] == 1
    assert comparison["available_prediction_records"] == 1
    assert comparison["primary_actor_output_samples"] == 1
    assert comparison["primary_actor_hits"] == 1
    assert comparison["primary_actor_hit_rate"] == 1.0
    assert comparison["human_consensus_hit_rate"] == 1.0
    assert comparison["strict_top1_accuracy"] == 1.0
    assert comparison["candidate_recall_at_k"] == 1.0
    assert comparison["disagreements"] == []
    assert comparison["non_evaluable"] == []

    # LG is allowed to output no final primary causal actor.  Such a sample is
    # excluded from the primary-actor hit-rate denominator, while its candidate
    # list can still contribute to Candidate Recall@K.
    no_primary = compare_predictions(
        consensus,
        [
            {
                "sample_id": samples[0]["sample_id"],
                "selected_actor_id": None,
                "candidate_actor_ids": [actor_id],
            }
        ],
        candidate_k=3,
    )
    assert no_primary["human_actor_consensus_samples"] == 1
    assert no_primary["available_prediction_records"] == 1
    assert no_primary["primary_actor_output_samples"] == 0
    assert no_primary["primary_actor_hits"] == 0
    assert no_primary["primary_actor_hit_rate"] != no_primary["primary_actor_hit_rate"]  # NaN
    assert no_primary["no_primary_actor_predictions"] == 1
    assert no_primary["candidate_recall_at_k"] == 1.0
    assert no_primary["disagreements"] == []
    assert len(no_primary["non_evaluable"]) == 1
    assert no_primary["sample_results"][0]["primary_actor_hit"] is None
    assert no_primary["sample_results"][0]["evaluation_status"] == "no_primary_actor_output"

    wrong_primary = compare_predictions(
        consensus,
        [
            {
                "sample_id": samples[0]["sample_id"],
                "selected_actor_id": "__OTHER_ACTOR__",
                "candidate_actor_ids": [actor_id],
            }
        ],
        candidate_k=3,
    )
    assert wrong_primary["primary_actor_output_samples"] == 1
    assert wrong_primary["primary_actor_hits"] == 0
    assert wrong_primary["primary_actor_hit_rate"] == 0.0
    assert len(wrong_primary["disagreements"]) == 1
    assert wrong_primary["non_evaluable"] == []

    # Building the Blocks tree catches most Gradio API incompatibilities without launching a server.
    from annotation_app import build_app

    app = build_app(str(CONFIG), "P03")
    assert app is not None
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
