"""Configuration for the interactive picker (loaded from a YAML file).

The same generator drives two deployments:

* **public** — every wheel index resolves to ``download.pytorch.org/whl/<channel>``.
* **internal** — the ``download.pytorch.org/whl/<channel>`` indexes are mirrored
  elsewhere (e.g. an Artifactory remote). Only channels listed under
  ``index_overrides`` are available; any other selected channel shows
  ``missing_index_message`` instead of an install command.

Only ``load()`` needs PyYAML, so importing this module (e.g. for the dataclass)
stays dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

DEFAULT_REPO_URL = "https://github.com/racinmat/pytorch-compatibility"
DEFAULT_TABLE_PATH = "data/COMPATIBILITY.md"
DEFAULT_INDEX_BASE = "https://download.pytorch.org/whl/"
DEFAULT_MISSING_MESSAGE = (
    "This wheel index has not been mirrored yet — please request it and clone "
    "the setup ticket to have it added."
)


@dataclass(frozen=True)
class PickerConfig:
    """Resolved picker configuration; embedded as JSON in the generated page."""

    mode: str = "public"  # "public" | "internal"
    default_index_base: str = DEFAULT_INDEX_BASE
    index_overrides: dict[str, str] = field(default_factory=dict)
    missing_index_message: str = DEFAULT_MISSING_MESSAGE
    repo_url: str = DEFAULT_REPO_URL
    table_url: str | None = None

    def resolved_table_url(self) -> str:
        if self.table_url:
            return self.table_url
        return self.repo_url.rstrip("/") + "/blob/master/" + DEFAULT_TABLE_PATH

    def to_payload(self) -> dict:
        """The subset the browser JS needs to resolve wheel-index URLs."""
        return {
            "mode": self.mode,
            "default_index_base": self.default_index_base,
            "index_overrides": dict(self.index_overrides),
            "missing_index_message": self.missing_index_message,
        }


def load(path: Path | str | None) -> PickerConfig:
    """Load config from a YAML file. Missing path/file -> built-in public defaults."""
    if path is None:
        return PickerConfig()
    p = Path(path)
    if not p.exists():
        return PickerConfig()

    import yaml  # local import so the rest of the package needs no third-party deps

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config {p} must be a YAML mapping, got {type(raw).__name__}.")

    known = {f.name for f in fields(PickerConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown config keys in {p}: {unknown}. Allowed: {sorted(known)}.")

    mode = raw.get("mode", "public")
    if mode not in ("public", "internal"):
        raise ValueError(f"config 'mode' must be 'public' or 'internal', got {mode!r}.")

    overrides = raw.get("index_overrides") or {}
    if not isinstance(overrides, dict):
        raise ValueError("config 'index_overrides' must be a mapping of channel -> url.")
    overrides = {str(k): str(v) for k, v in overrides.items()}

    return PickerConfig(
        mode=mode,
        default_index_base=str(raw.get("default_index_base", DEFAULT_INDEX_BASE)),
        index_overrides=overrides,
        missing_index_message=str(raw.get("missing_index_message", DEFAULT_MISSING_MESSAGE)),
        repo_url=str(raw.get("repo_url", DEFAULT_REPO_URL)),
        table_url=raw.get("table_url"),
    )
