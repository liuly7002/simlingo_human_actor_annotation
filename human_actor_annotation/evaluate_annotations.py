from __future__ import annotations

import argparse
import json
from pathlib import Path

from annotation_core.config import load_config
from annotation_core.dataset import load_jsonl
from annotation_core.metrics import (
    build_consensus,
    compare_predictions,
    krippendorff_alpha_binary,
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
    parser = argparse.ArgumentParser(description="汇总人工关注对象标注并与 LG 结果比较")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--predictions",
        default=None,
        help=(
            "可选：LG 结果 JSON/JSONL。至少包含 sample_id；LG 实际输出主要 actor 时"
            "计算其是否落入人工共识集合，如包含候选 actor 列表还会计算 Candidate Recall@K"
        ),
    )
    parser.add_argument(
        "--consensus-threshold",
        type=float,
        default=0.5,
        help="人工 actor 进入共识集合所需的严格投票比例阈值，默认 > 0.5",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=3,
        help="评价 LG 分析候选召回率时使用的 K，默认 3",
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
    manifest_samples = load_jsonl(config.dataset.manifest_path)
    consensus = build_consensus(records, threshold=args.consensus_threshold)
    agreement = pairwise_agreement(records)
    agreement["krippendorff_alpha_binary"] = krippendorff_alpha_binary(
        records, manifest_samples
    )
    agreement["annotation_records"] = len(records)
    agreement["unique_samples"] = len(
        {str(record.get("sample_id", "")) for record in records}
    )
    agreement["annotators"] = sorted(
        {str(record.get("annotator_id", "")) for record in records}
    )

    write_csv(output_dir / "consensus.csv", consensus)
    comparison = None
    if args.predictions:
        prediction_records = read_json_or_jsonl(
            Path(args.predictions).expanduser().resolve()
        )
        comparison = compare_predictions(
            consensus,
            prediction_records,
            candidate_k=args.candidate_k,
        )
        sample_results = comparison.pop("sample_results")
        disagreements = comparison.pop("disagreements")
        non_evaluable = comparison.pop("non_evaluable")
        write_csv(
            output_dir / "prediction_comparison.csv",
            sample_results,
        )
        write_csv(
            output_dir / "prediction_disagreements.csv",
            disagreements,
        )
        write_csv(
            output_dir / "prediction_non_evaluable.csv",
            non_evaluable,
        )

    summary = {
        "agreement": {key: json_number(value) for key, value in agreement.items()},
        "consensus_threshold": args.consensus_threshold,
        "consensus_samples": len(consensus),
        "actor_consensus_samples": sum(
            item.get("consensus_status") == "actors" for item in consensus
        ),
        "no_attention_consensus_samples": sum(
            item.get("consensus_status") == "none" for item in consensus
        ),
        "needs_adjudication": sum(
            bool(item.get("needs_adjudication")) for item in consensus
        ),
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
