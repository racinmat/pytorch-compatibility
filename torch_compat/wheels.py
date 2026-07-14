"""Parse the official PyTorch wheel index into structured rows.

Source of truth: ``https://download.pytorch.org/whl/torch/`` — a PEP 503 style HTML
listing of every ``torch`` wheel ever published. Each wheel filename encodes, in a
fully deterministic way:

    torch-<version>[+<local>]-<pytag>-<abitag>-<platform>.whl

from which we extract the torch version, the compute backend (cpu / cuXYZ / rocm / xpu),
the supported Python version, and the target platform.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from urllib.parse import unquote

from . import sources

WHEEL_INDEX_URL = "https://download.pytorch.org/whl/torch/"
# PyPI JSON API: authoritative list of what is published to PyPI itself (as opposed to
# the download.pytorch.org alternative registry). PyPI rejects local version tags, so
# only the untagged "default" build can appear here.
PYPI_JSON_URL = "https://pypi.org/pypi/torch/json"

# torch-2.3.0+cu121-cp311-cp311-linux_x86_64.whl
# torch-2.0.0-1-cp38-cp38-...whl  (optional PEP 427 build tag "-1")
_WHEEL_RE = re.compile(
    r"^torch-"
    r"(?P<version>\d[^+-]*)"           # 2.3.0 / 0.3.0.post4 / 2.6.0.dev20240101
    r"(?:\+(?P<local>[^-]+))?"          # +cu121 / +cpu / +rocm6.0 / +xpu  (optional)
    r"(?:-(?P<build>\d[^-]*))?"          # optional build tag (starts with a digit)
    r"-(?P<pytag>[A-Za-z][^-]*)"       # cp311 / cp313t / cp27 / py3 (starts with a letter)
    r"-(?P<abitag>[^-]+)"              # cp311 / cp27mu / none / abi3
    r"-(?P<platform>.+)"              # linux_x86_64 / macosx_11_0_arm64 / manylinux...
    r"\.whl$"
)


@dataclass(frozen=True)
class WheelRow:
    torch_version: str
    backend: str        # "cpu", "cuda", "rocm", "xpu", or "default"
    backend_version: str  # "12.1" for cuda/rocm; "" otherwise
    python: str         # "3.11"
    python_tag: str     # "cp311" / "cp313t"
    os: str             # "linux" / "windows" / "macos"
    arch: str           # "x86_64" / "aarch64" / "arm64" / ...
    filename: str
    upload_date: str    # "2024-04-24" (from the index data-upload-time), or ""

    def as_dict(self) -> dict:
        return asdict(self)


def _parse_python(pytag: str) -> str:
    m = re.match(r"cp(\d)(\d+)", pytag)
    if not m:
        # e.g. "py3", "py2.py3" -> unknown precise version
        return pytag
    return f"{m.group(1)}.{m.group(2)}"


def _parse_backend(local: str | None) -> tuple[str, str]:
    if not local:
        return "default", ""
    if local == "cpu":
        return "cpu", ""
    if local == "xpu":
        return "xpu", ""
    m = re.match(r"cu(\d+)$", local)
    if m:
        digits = m.group(1)
        return "cuda", f"{digits[:-1]}.{digits[-1]}"
    m = re.match(r"rocm([\d.]+)$", local)
    if m:
        return "rocm", m.group(1)
    return local, ""


def _parse_platform(platform: str) -> tuple[str, str]:
    p = platform.lower()
    if p.startswith("win"):
        arch = "arm64" if "arm64" in p else "x86_64"
        return "windows", arch
    if p.startswith("macosx"):
        arch = "arm64" if "arm64" in p else "x86_64"
        return "macos", arch
    # linux / manylinux variants
    for arch in ("x86_64", "aarch64", "s390x", "ppc64le", "arm64"):
        if arch in p:
            return "linux", arch
    return "linux", "unknown"


_ANCHOR_RE = re.compile(r"<a\s*(?P<attrs>[^>]*)>(?P<name>[^<>]+\.whl)</a>")
_UPLOAD_RE = re.compile(r'data-upload-time="(?P<ts>[^"]*)"')


def _iter_wheels(html: str):
    """Yield (filename, upload_date) for each wheel anchor in the index HTML."""
    for m in _ANCHOR_RE.finditer(html):
        name = unquote(m.group("name"))
        ts = _UPLOAD_RE.search(m.group("attrs"))
        upload_date = ts.group("ts")[:10] if ts else ""  # ISO date part (YYYY-MM-DD)
        yield name, upload_date


def parse_index(html: str) -> list[WheelRow]:
    rows: list[WheelRow] = []
    seen: set[str] = set()
    for name, upload_date in _iter_wheels(html):
        if name in seen:
            continue
        seen.add(name)
        m = _WHEEL_RE.match(name)
        if not m:
            continue
        backend, backend_version = _parse_backend(m.group("local"))
        os_name, arch = _parse_platform(m.group("platform"))
        rows.append(
            WheelRow(
                torch_version=m.group("version"),
                backend=backend,
                backend_version=backend_version,
                python=_parse_python(m.group("pytag")),
                python_tag=m.group("pytag"),
                os=os_name,
                arch=arch,
                filename=name,
                upload_date=upload_date,
            )
        )
    return rows


def load_wheels(*, force: bool = False) -> list[WheelRow]:
    html = sources.fetch(WHEEL_INDEX_URL, force=force)
    return parse_index(html)


def parse_pypi_filenames(payload: str) -> set[str]:
    """Return the set of wheel filenames published to PyPI (from the JSON API payload)."""
    data = json.loads(payload)
    names: set[str] = set()
    for files in data.get("releases", {}).values():
        for f in files:
            fn = f.get("filename", "")
            if fn.endswith(".whl"):
                names.add(fn)
    return names


def load_pypi_data(*, force: bool = False) -> dict:
    return json.loads(sources.fetch(PYPI_JSON_URL, force=force))


def load_pypi_filenames(*, force: bool = False) -> set[str]:
    return parse_pypi_filenames(sources.fetch(PYPI_JSON_URL, force=force))


# The CUDA version of the default `pip install torch` build is not in the filename, but the
# wheel METADATA pins the matching NVIDIA CUDA packages, which reveal the exact minor version.
# Two historical formats: `... (==12.4.127)` (<=2.6) and `...==12.6.77` / `cuda-toolkit[...]==13.0.3`.
_CUDA_DEP_PATTERNS = (
    r"nvidia-cuda-runtime-cu\d+\s*\(?==(\d+\.\d+)",
    r"cuda-toolkit\[[^\]]*\]\s*\(?==(\d+\.\d+)",
    r"nvidia-cuda-nvrtc-cu\d+\s*\(?==(\d+\.\d+)",
)


def parse_default_cuda(metadata_text: str) -> str:
    """Extract the CUDA minor version (e.g. "13.0") from a wheel's Requires-Dist metadata."""
    for pat in _CUDA_DEP_PATTERNS:
        m = re.search("Requires-Dist: " + pat, metadata_text)
        if m:
            return m.group(1)
    return ""


def load_pypi_default_cuda(versions, *, force: bool = False) -> dict[str, str]:
    """Map torch version -> CUDA minor of the default Linux x86_64 PyPI wheel.

    Reads each version's linux-x86_64 wheel METADATA (PEP 658 ``.metadata`` sidecar) and
    parses the pinned NVIDIA CUDA dependency. Versions predating the nvidia-* pip deps
    (older 1.x) resolve to "" and are simply left unmarked.
    """
    data = load_pypi_data(force=force)
    out: dict[str, str] = {}
    for v in versions:
        cand = next(
            (
                f
                for f in data.get("releases", {}).get(v, [])
                if re.search(r"cp3\d+-cp3\d+-manylinux.*x86_64", f.get("filename", ""))
            ),
            None,
        )
        if not cand or not cand.get("url"):
            continue
        meta = sources.try_fetch(cand["url"] + ".metadata", force=force)
        if not meta:
            continue
        cu = parse_default_cuda(meta)
        if cu:
            out[v] = cu
    return out
