from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from annotation_core.config import load_config
from annotation_core.dataset import load_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="将标注样本均衡分配给多名测试人员")
    parser.add_argument("--config", required=True)
    parser.add_argument("--annotators", nargs="+", required=True, help="例如 P01 P02 P03 P04")
    parser.add_argument(
        "--redundancy",
        type=int,
        default=3,
        help="每条样本由多少名测试人员独立标注",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    samples = load_jsonl(config.dataset.manifest_path)
    annotators = list(dict.fromkeys(args.annotators))
    if not annotators:
        raise ValueError("annotators 不能为空")
    if args.redundancy < 1 or args.redundancy > len(annotators):
        raise ValueError("redundancy 必须在 1 到标注员数量之间")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else config.dataset.work_dir / "assignments"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.sampling.random_seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    assignments = {annotator: [] for annotator in annotators}

    for sample_index, sample in enumerate(shuffled):
        # Rotating contiguous blocks produce a balanced, reproducible assignment.
        start = sample_index % len(annotators)
        chosen = [annotators[(start + offset) % len(annotators)] for offset in range(args.redundancy)]
        for annotator in chosen:
            assignments[annotator].append(sample)

    summary = {
        "total_samples": len(samples),
        "annotators": annotators,
        "redundancy": args.redundancy,
        "assignments": {},
    }
    for annotator, items in assignments.items():
        path = output_dir / f"manifest_{annotator}.jsonl"
        write_jsonl(path, items)
        summary["assignments"][annotator] = {"samples": len(items), "manifest": str(path)}

    with (output_dir / "assignment_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
