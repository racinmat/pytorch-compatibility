"""HTTP fetching with on-disk caching.

Every raw upstream artifact (wheel index HTML, build scripts) is cached under
``data/raw/`` so that a full run is reproducible offline and the exact inputs used
to produce a table are auditable. Delete the cache (or pass ``force=True``) to refresh.
"""

from __future__ import annotations

import hashlib
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

_USER_AGENT = "torch-compat/1.0 (+deterministic compatibility table generator)"


def _cache_path(url: str, cache_dir: Path) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    # Keep a human-readable hint plus a hash to avoid collisions / illegal chars.
    hint = url.rstrip("/").split("/")[-1] or "index"
    hint = "".join(c if c.isalnum() or c in "._-" else "_" for c in hint)[:60]
    return cache_dir / f"{hint}.{digest}"


def _missing_path(path: Path) -> Path:
    return path.with_name(path.name + ".missing")


def _download(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (trusted hosts)
        return resp.read().decode("utf-8", errors="replace")


def fetch(url: str, *, cache_dir: Path | None = None, force: bool = False) -> str:
    """Return the text body of ``url``, using an on-disk cache.

    Raises ``urllib.error.HTTPError`` for non-2xx responses (callers may catch 404).
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(url, cache_dir)
    if path.exists() and not force:
        return path.read_text(encoding="utf-8")
    body = _download(url)
    path.write_text(body, encoding="utf-8")
    _missing_path(path).unlink(missing_ok=True)
    return body


def try_fetch(url: str, *, cache_dir: Path | None = None, force: bool = False) -> str | None:
    """Like :func:`fetch` but return ``None`` on a 404/URL error instead of raising.

    Both hits and misses are cached (negative caching), so repeated runs never re-probe
    the many 404 candidate URLs and can run fully offline.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(url, cache_dir)
    missing = _missing_path(path)
    if not force:
        if path.exists():
            return path.read_text(encoding="utf-8")
        if missing.exists():
            return None
    try:
        body = _download(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            missing.write_text("", encoding="utf-8")
            return None
        raise
    except urllib.error.URLError:
        return None
    path.write_text(body, encoding="utf-8")
    missing.unlink(missing_ok=True)
    return body
