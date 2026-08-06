from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from finassist.config import get_app_config, project_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FinAssist FastAPI service.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = project_root() / config_path
        os.environ["FINASSIST_CONFIG_PATH"] = str(config_path.resolve())
        get_app_config.cache_clear()

    uvicorn.run(
        "finassist.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=False,
    )


if __name__ == "__main__":
    main()
