from __future__ import annotations

import argparse
import json
from pathlib import Path

from annotation_core.config import load_config
from annotation_core.metrics import (
    build_consensus,
    compare_predictions,
    fleiss_kappa,
    pairwise_agreement,
    read_annotation_dir,
    read_json_or_jsonl,
    write_csv,
)


def json_number(value):
    try:
        if value != value:  # NaN
            return None
    except Exception:
        pass
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总人工标注一致性并与 LG 预测比较")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--predictions",
        default=None,
        help="可选：方法预测 JSON/JSONL，至少包含 sample_id 与 selected_actor_id",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else config.dataset.work_dir / "evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_annotation_dir(config.dataset.annotation_dir)
    consensus = build_consensus(records)
    agreement = pairwise_agreement(records)
    agreement["fleiss_kappa"] = fleiss_kappa(records)
    agreement["annotation_records"] = len(records)
    agreement["unique_samples"] = len({str(r.get('sample_id', '')) for r in records})
    agreement["annotators"] = sorted({str(r.get('annotator_id', '')) for r in records})

    write_csv(output_dir / "consensus.csv", consensus)
    comparison = None
    if args.predictions:
        prediction_records = read_json_or_jsonl(Path(args.predictions).expanduser().resolve())
        comparison = compare_predictions(consensus, prediction_records)
        write_csv(output_dir / "prediction_disagreements.csv", comparison.pop("disagreements"))

    summary = {
        "agreement": {key: json_number(value) for key, value in agreement.items()},
        "consensus_samples": len(consensus),
        "needs_adjudication": sum(bool(item.get("needs_adjudication")) for item in consensus),
        "prediction_comparison": (
            {key: json_number(value) for key, value in comparison.items()}
            if comparison is not None
            else None
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n结果目录：{output_dir}")


if __name__ == "__main__":
    main()
