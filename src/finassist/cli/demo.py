from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _demo_app_path() -> Path:
    import finassist.demo.app as demo_module

    return Path(demo_module.__file__).resolve()


def main() -> None:
    if "FINASSIST_CONFIG_PATH" not in os.environ:
        from finassist.config import project_root

        os.environ["FINASSIST_CONFIG_PATH"] = str(
            (project_root() / "configs" / "offline.yaml").resolve()
        )

    app_path = _demo_app_path()
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
