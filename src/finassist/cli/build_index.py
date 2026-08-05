from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the vector index from processed chunks (Phase 2 scaffold)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (defaults to configs/default.yaml).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    parser.parse_args()
    raise SystemExit(
        "Index building is not implemented yet. Run `finassist-ingest` first, "
        "then implement `src/finassist/index/` in the next phase."
    )


if __name__ == "__main__":
    main()
