"""Combine wheel metadata and build-script arch lists into a compatibility table."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from . import arch_lists, wheels
from .wheels import WheelRow


def version_key(v: str) -> tuple:
    nums = [int(x) for x in re.findall(r"\d+", v)[:4]]
    while len(nums) < 4:
        nums.append(0)
    # dev/rc/a/b sort before final release of same x.y.z
    pre = 0 if re.search(r"(dev|rc|a\d|b\d|alpha|beta)", v) else 1
    return (nums[0], nums[1], nums[2], pre, nums[3])


def python_key(p: str) -> tuple:
    nums = [int(x) for x in re.findall(r"\d+", p)]
    return tuple(nums) if nums else (0,)


@dataclass
class CompatRow:
    torch_version: str
    release_date: str  # earliest wheel upload date for this torch version (YYYY-MM-DD)
    backend: str
    backend_version: str
    python_min: str
    python_max: str
    pythons: list[str]
    compute_capabilities: list[str]
    arch_list_raw: str
    has_ptx: bool
    platforms: list[str]
    on_pypi: bool          # this exact build is downloadable from PyPI
    version_on_pypi: bool   # any build of this torch version is on PyPI (else alt registry only)
    pypi_linux_default: bool  # this CUDA build is what a bare `pip install torch` gives on Linux
    arch_source_url: str


def build_rows(
    wheel_rows: list[WheelRow],
    *,
    pypi_filenames: set[str] | None = None,
    pypi_default_cuda: dict[str, str] | None = None,
    force: bool = False,
    verbose: bool = False,
) -> list[CompatRow]:
    pypi_filenames = pypi_filenames or set()
    pypi_default_cuda = pypi_default_cuda or {}
    groups: dict[tuple[str, str, str], dict] = defaultdict(
        lambda: {"pythons": set(), "platforms": set(), "on_pypi": False}
    )
    # Release date of a torch version = earliest wheel upload date across all its wheels.
    release_date: dict[str, str] = {}
    version_on_pypi: dict[str, bool] = defaultdict(bool)
    for w in wheel_rows:
        key = (w.torch_version, w.backend, w.backend_version)
        groups[key]["pythons"].add(w.python)
        groups[key]["platforms"].add(f"{w.os}/{w.arch}")
        on_pypi = w.filename in pypi_filenames
        groups[key]["on_pypi"] = groups[key]["on_pypi"] or on_pypi
        version_on_pypi[w.torch_version] = version_on_pypi[w.torch_version] or on_pypi
        if w.upload_date:
            prev = release_date.get(w.torch_version)
            if prev is None or w.upload_date < prev:
                release_date[w.torch_version] = w.upload_date

    arch_cache: dict[tuple[str, str], arch_lists.ArchResult | None] = {}
    rows: list[CompatRow] = []
    for (torch_version, backend, backend_version), data in groups.items():
        caps: list[str] = []
        arch_raw = ""
        has_ptx = False
        source_url = ""
        if backend == "cuda":
            ck = (torch_version, backend_version)
            if ck not in arch_cache:
                if verbose:
                    print(f"  arch list: torch {torch_version} cuda {backend_version}")
                arch_cache[ck] = arch_lists.arch_result(
                    torch_version, backend_version, force=force
                )
            res = arch_cache[ck]
            if res is not None:
                caps = list(res.capabilities)
                arch_raw = res.arch_list_raw
                has_ptx = res.has_ptx
                source_url = res.source_url

        pys = sorted(data["pythons"], key=python_key)
        rows.append(
            CompatRow(
                torch_version=torch_version,
                release_date=release_date.get(torch_version, ""),
                backend=backend,
                backend_version=backend_version,
                python_min=pys[0] if pys else "",
                python_max=pys[-1] if pys else "",
                pythons=pys,
                compute_capabilities=caps,
                arch_list_raw=arch_raw,
                has_ptx=has_ptx,
                platforms=sorted(data["platforms"]),
                on_pypi=data["on_pypi"],
                version_on_pypi=version_on_pypi[torch_version],
                pypi_linux_default=(
                    backend == "cuda"
                    and backend_version == pypi_default_cuda.get(torch_version)
                    and any(p.startswith("linux/") for p in data["platforms"])
                ),
                arch_source_url=source_url,
            )
        )

    rows.sort(
        key=lambda r: (
            version_key(r.torch_version),
            r.backend,
            version_key(r.backend_version or "0"),
        ),
        reverse=False,
    )
    return rows


# --- rendering --------------------------------------------------------------

def write_json(rows: list[CompatRow], path: Path) -> None:
    path.write_text(
        json.dumps([asdict(r) for r in rows], indent=2) + "\n", encoding="utf-8"
    )


def write_csv(rows: list[CompatRow], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "torch_version",
                "release_date",
                "backend",
                "backend_version",
                "python_min",
                "python_max",
                "pythons",
                "compute_capabilities",
                "arch_list_raw",
                "has_ptx",
                "platforms",
                "on_pypi",
                "version_on_pypi",
                "pypi_linux_default",
                "arch_source_url",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.torch_version,
                    r.release_date,
                    r.backend,
                    r.backend_version,
                    r.python_min,
                    r.python_max,
                    " ".join(r.pythons),
                    ";".join(r.compute_capabilities),
                    r.arch_list_raw,
                    r.has_ptx,
                    " ".join(r.platforms),
                    r.on_pypi,
                    r.version_on_pypi,
                    r.pypi_linux_default,
                    r.arch_source_url,
                ]
            )


def _py_range(pys: list[str]) -> str:
    if not pys:
        return "-"
    if len(pys) == 1:
        return pys[0]
    return f"{pys[0]} – {pys[-1]}"


def _is_dev(version: str) -> bool:
    return "dev" in version.lower()


def write_markdown(rows: list[CompatRow], path: Path) -> None:
    cuda_rows = [r for r in rows if r.backend == "cuda" and not _is_dev(r.torch_version)]
    cuda_rows.sort(
        key=lambda r: (version_key(r.torch_version), version_key(r.backend_version)),
        reverse=True,
    )

    lines: list[str] = []
    lines.append("# PyTorch compatibility table")
    lines.append("")
    lines.append(
        "Generated deterministically from the official wheel index and PyTorch build "
        "scripts. Regenerate with `python generate.py`. Do not edit by hand."
    )
    lines.append("")
    lines.append("## CUDA builds")
    lines.append("")
    lines.append(
        "`Compute capabilities` are the SM architectures compiled into the wheel "
        "(`TORCH_CUDA_ARCH_LIST`). `+PTX` means PTX is embedded for the highest arch, "
        "giving forward compatibility with newer GPUs via JIT."
    )
    lines.append("")
    lines.append(
        "`pip default (Linux)` marks the single CUDA build you get from a bare "
        "`pip install torch` on Linux x86_64 (determined from the PyPI wheel's pinned NVIDIA "
        "CUDA dependency). Every other CUDA build is only on the `download.pytorch.org` "
        "alternative registry and needs "
        "`pip install torch --index-url https://download.pytorch.org/whl/cuXXX`."
    )
    lines.append("")
    header = [
        "torch",
        "Released",
        "pip default (Linux)",
        "CUDA",
        "Python",
        "Compute capabilities",
        "PTX",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for r in cuda_rows:
        caps = ", ".join(r.compute_capabilities) if r.compute_capabilities else "?"
        lines.append(
            "| "
            + " | ".join(
                [
                    r.torch_version,
                    r.release_date or "-",
                    "yes" if r.pypi_linux_default else "",
                    r.backend_version,
                    _py_range(r.pythons),
                    caps,
                    "yes" if r.has_ptx else "",
                ]
            )
            + " |"
        )

    # Python support summary (per torch version, across all backends)
    lines.append("")
    lines.append("## Python support per torch version")
    lines.append("")
    per_torch: dict[str, set[str]] = defaultdict(set)
    released: dict[str, str] = {}
    on_pypi: dict[str, bool] = {}
    for r in rows:
        if _is_dev(r.torch_version):
            continue
        per_torch[r.torch_version].update(r.pythons)
        released[r.torch_version] = r.release_date
        on_pypi[r.torch_version] = r.version_on_pypi
    lines.append("| torch | Released | On PyPI | Python |")
    lines.append("| --- | --- | --- | --- |")
    for tv in sorted(per_torch, key=version_key, reverse=True):
        pys = sorted(per_torch[tv], key=python_key)
        flag = "yes" if on_pypi.get(tv) else "no"
        lines.append(f"| {tv} | {released.get(tv) or '-'} | {flag} | {_py_range(pys)} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate(out_dir: Path, *, force: bool = False, verbose: bool = False) -> list[CompatRow]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print("Fetching wheel index and PyPI file list ...")
    wheel_rows = wheels.load_wheels(force=force)
    pypi_filenames = wheels.load_pypi_filenames(force=force)
    # Versions with a Linux CUDA build that are on PyPI: resolve their default pip-install CUDA.
    cuda_linux_versions = {
        w.torch_version
        for w in wheel_rows
        if w.backend == "cuda" and w.os == "linux" and w.arch == "x86_64"
    }
    if verbose:
        print(f"Resolving default pip-install CUDA for {len(cuda_linux_versions)} versions ...")
    pypi_default_cuda = wheels.load_pypi_default_cuda(cuda_linux_versions, force=force)
    if verbose:
        print(f"Parsed {len(wheel_rows)} wheels. Resolving CUDA arch lists ...")
    rows = build_rows(
        wheel_rows,
        pypi_filenames=pypi_filenames,
        pypi_default_cuda=pypi_default_cuda,
        force=force,
        verbose=verbose,
    )
    write_json(rows, out_dir / "compatibility_table.json")
    write_csv(rows, out_dir / "compatibility_matrix.csv")
    write_markdown(rows, out_dir / "COMPATIBILITY.md")
    return rows
