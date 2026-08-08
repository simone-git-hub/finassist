from __future__ import annotations

import csv
import hashlib
import warnings
from datetime import UTC, datetime
from pathlib import Path

import httpx

from finassist.config import AppConfig, project_root
from finassist.models import ManifestEntry


def load_manifest(manifest_path: Path | None = None) -> list[ManifestEntry]:
    path = manifest_path or project_root() / "data" / "raw" / "manifest.csv"
    entries: list[ManifestEntry] = []

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            entries.append(ManifestEntry.model_validate(row))

    return entries


def _cache_path(url: str, cache_dir: Path) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.html"


def fetch_local_html(local_path: Path) -> tuple[str, datetime, bool]:
    if not local_path.exists():
        raise FileNotFoundError(f"Local HTML file not found: {local_path}")
    fetched_at = datetime.fromtimestamp(local_path.stat().st_mtime, tz=UTC)
    return local_path.read_text(encoding="utf-8"), fetched_at, True


def fetch_url(
    url: str,
    *,
    cache_dir: Path,
    timeout_s: float = 20.0,
    user_agent: str = "FinAssistBot/0.1",
    force_refresh: bool = False,
    local_path: Path | None = None,
) -> tuple[str, datetime, bool]:
    """Fetch HTML for a URL with on-disk caching.

    If `local_path` is provided, read from disk instead of HTTP (offline dev).
    Returns `(html, fetched_at, from_cache)`.
    """
    if local_path is not None:
        return fetch_local_html(local_path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(url, cache_dir)

    if cache_file.exists() and not force_refresh:
        fetched_at = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=UTC)
        return cache_file.read_text(encoding="utf-8"), fetched_at, True

    headers = {"User-Agent": user_agent}
    try:
        with httpx.Client(timeout=timeout_s, follow_redirects=True, headers=headers) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        if cache_file.exists():
            warnings.warn(
                f"Fetch failed for {url} ({exc}); using cached HTML.",
                stacklevel=2,
            )
            fetched_at = datetime.fromtimestamp(cache_file.stat().st_mtime, tz=UTC)
            return cache_file.read_text(encoding="utf-8"), fetched_at, True
        raise

    html = response.text
    cache_file.write_text(html, encoding="utf-8")
    fetched_at = datetime.now(tz=UTC)
    return html, fetched_at, False


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def resolve_ingest_paths(config: AppConfig, root: Path | None = None) -> tuple[Path, Path, Path]:
    base = root or project_root()
    manifest_path = base / config.ingest.manifest_path
    cache_dir = base / config.ingest.cache_dir
    processed_dir = base / config.ingest.processed_dir
    return manifest_path, cache_dir, processed_dir
