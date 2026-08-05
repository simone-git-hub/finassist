from __future__ import annotations

import argparse

from finassist.config import get_app_config
from finassist.ingest.pipeline import run_ingest_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch manifest URLs, parse HTML, chunk text, and write processed JSONL."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (defaults to configs/default.yaml).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-download pages even if cached HTML exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N manifest rows (useful for smoke tests).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Optional manifest CSV path (relative to project root).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = get_app_config(args.config) if args.config else get_app_config()
    if args.manifest:
        config.ingest.manifest_path = args.manifest

    documents, chunks, documents_path, chunks_path = run_ingest_pipeline(
        config,
        force_refresh=args.force_refresh,
        limit=args.limit,
    )

    print(f"Ingested {len(documents)} documents -> {documents_path}")
    print(f"Created {len(chunks)} chunks -> {chunks_path}")


if __name__ == "__main__":
    main()
