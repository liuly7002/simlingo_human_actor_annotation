from __future__ import annotations

import argparse
import json

from annotation_core.config import load_config
from annotation_core.dataset import prepare_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="从 SimLingo 数据集生成盲评标注 manifest")
    parser.add_argument("--config", required=True, help="YAML 配置文件")
    args = parser.parse_args()

    config = load_config(args.config)
    stats = prepare_manifest(config)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nManifest: {config.dataset.manifest_path}")
    print(f"Summary : {config.dataset.work_dir / 'prepare_summary.json'}")


if __name__ == "__main__":
    main()
