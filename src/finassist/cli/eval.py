from __future__ import annotations

import argparse
import os
from pathlib import Path

from finassist.config import get_app_config, project_root
from finassist.eval.runner import format_report_table, run_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run retrieval and RAG evaluation on gold Q&A.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--gold-path",
        type=str,
        default=None,
        help="Path to gold JSONL (default from config eval.gold_path).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for JSON results (default: results/).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = project_root() / config_path
        os.environ["FINASSIST_CONFIG_PATH"] = str(config_path.resolve())
        get_app_config.cache_clear()

    config = get_app_config()
    gold_path = Path(args.gold_path) if args.gold_path else project_root() / config.eval.gold_path
    output_dir = Path(args.output_dir) if args.output_dir else None

    report = run_evaluation(config, gold_path=gold_path, output_dir=output_dir)
    print(format_report_table(report))


if __name__ == "__main__":
    main()
