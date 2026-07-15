# [pytorch-compatibility](https://racinmat.github.io/pytorch-compatibility/)

Two things in one small, dependency-free project:

1. An **interactive install-command picker** ([`docs/index.html`](docs/index.html)) — choose a
   PyTorch version, OS, compute platform (or just your **GPU**) and Python version, and get
   ready-to-paste commands for **pip, uv, Poetry and PDM**.
2. A **generator** that builds a **PyTorch × CUDA × compute-capability × Python** compatibility
   table **deterministically** from upstream metadata and source code — no manual transcription,
   no scraping of prose docs. The picker is generated from the same data.

Rerunning the generator reproduces everything exactly, and newer PyTorch/CUDA releases are picked
up automatically without code changes.

## The interactive picker

Open [`docs/index.html`](docs/index.html) directly in a browser — it is fully self-contained
(all data is embedded as JSON), so it needs no server and no network. Its header links straight to
this **GitHub repo** and to the rendered **[compatibility table](data/COMPATIBILITY.md)**, and the
page has two tabs:

- **Search by Python** — the install-command picker described below (start from your Python version
  and GPU, get the exact command).
- **Search by PyTorch** — start from a **torch version** and see everything it ships: available
  **Python** versions, every **CUDA** build with its **compute capabilities** and the matching
  **NVIDIA GPUs**, and every **ROCm** build with its **gfx targets** and matching **AMD GPUs**.

### Search by Python

- Pick your **Python version** and **GPU vendor** (NVIDIA / AMD), then **search for your GPU** in a
  filterable dropdown (by name/model/gfx, e.g. `4090`, `A100`, `7900 XTX`, `MI300X`). NVIDIA cards
  offer only **CUDA / CPU** platforms; AMD cards offer only **ROCm / CPU**.
- Selecting a GPU infers its compatibility and auto-selects a build. Only builds your GPU can
  actually run are shown — the rest are hidden (there can be many). For NVIDIA, CUDA builds needing
  JIT are flagged **PTX**; for AMD, matching uses each card's ROCm support window.
- Then pick the **torch version** (only versions supporting your Python are offered), **OS**,
  **compute platform** and **package manager**.
- A note spells out the compute-capability coverage of each torch version. Because the arch list
  **differs per CUDA build**, it distinguishes the floor of the plain `pip install` (Linux default)
  wheel from the older CUDA build you must install explicitly to reach the oldest cards, and it
  applies CUDA's real compatibility rules: newer same-major GPUs run via **binary
  forward-compatibility** (e.g. CC 12.1 / NVIDIA DGX Spark on the CC 12.0 build), and `+PTX` builds
  JIT-compile for even newer architectures — only a strictly newer architecture *major* needs a
  later torch release. For AMD it shows the **ROCm version range** instead. This makes clear that
  cards outside the range are intentionally unavailable, not a bug.
- Every inferred fact links to its **source**: the GPU's compute capability / gfx target
  ([NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda/gpus) /
  [legacy](https://developer.nvidia.com/cuda/gpus/legacy),
  [AMD ROCm support matrix](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html),
  [LLVM gfx targets](https://llvm.org/docs/AMDGPUUsage.html#processors)) and the selected torch
  release on GitHub, its [PyPI project page](https://pypi.org/project/torch/), and the exact
  `download.pytorch.org/whl/<channel>` wheel index the chosen platform installs from.
- Choose your **package manager** (pip / uv / Poetry / PDM) and get its exact command with a copy
  button — including the extra `source` / `[[tool.pdm.source]]` / `[[tool.uv.index]]` config that
  Poetry, PDM and uv projects need to reach the non-default `download.pytorch.org` index.

Compatibility uses CUDA **binary minor-version forward compatibility** (a `sm_8x` cubin runs on a
same-major GPU with an equal-or-higher minor, e.g. an `8.6` binary runs on an `8.9` RTX 4090) plus
**PTX JIT forward compatibility** (`+PTX` builds run on newer GPUs).

### Which AMD GPUs work with PyTorch ROCm (and why some don't)

PyTorch's prebuilt ROCm wheels are compiled for a **fixed set of AMD GPU ISAs** (`gfx*` targets in
`PYTORCH_ROCM_ARCH`), and ROCm's HIP runtime + math libraries (rocBLAS/MIOpen) only ship kernels
for those. A card whose `gfx` target isn't in the wheel has **no usable binary** — so the picker
lists only officially supported cards.

- **Radeon RX 5700 / 5700 XT / 5600 / 5500 (RDNA1, `gfx1010`) are not supported.** RDNA1 was never
  an official ROCm compute target, so no PyTorch ROCm wheel is built for it. The common
  `HSA_OVERRIDE_GFX_VERSION=10.3.0` "pretend it's an RX 6800 (`gfx1030`)" trick is unofficial,
  unsupported and unreliable, so these cards are intentionally omitted.
- **Polaris and older (RX 400/500 `gfx803`, and earlier GCN) are not supported** — only ever
  experimental in very old ROCm, dropped long ago.
- **Oldest cards that do work:** `gfx900` (RX Vega 56/64) on ROCm ≤ 5.4 and `gfx906`
  (Radeon VII, Instinct MI50/MI60) on ROCm ≤ 5.7 — i.e. only with older torch + ROCm builds. On
  **current ROCm 6.x** the oldest usable are **Instinct MI100** (`gfx908`, datacenter) and the
  **Radeon RX 6800 / RDNA2** family (`gfx1030`, consumer/workstation).

In short: for a recent PyTorch, the practical minimum AMD consumer GPU is an **RX 6800 (RDNA2)**;
anything older than RDNA2 needs an older torch/ROCm, and RDNA1 (RX 5000) isn't supported at all.

**Sources — go straight to the primary references:**

- [ROCm compatibility matrix](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
  — the authoritative per-ROCm-version list of supported GPUs / architectures.
- [Use ROCm on Radeon GPUs — compatibility matrix](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility/native_linux/native_linux_compatibility.html)
  — the consumer/Radeon (RDNA) support list (Windows Subsystem for Linux and native).
- [ROCm hardware/GPU support & OS matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
  — architecture ↔ gfx target ↔ product mapping.
- [PyTorch – Get Started (select ROCm)](https://pytorch.org/get-started/locally/) and
  [previous versions](https://pytorch.org/get-started/previous-versions/) — the official install
  commands and which ROCm each torch ships.
- [`PYTORCH_ROCM_ARCH` in the manywheel build scripts](https://github.com/pytorch/pytorch/blob/main/.ci/manywheel/build_rocm.sh)
  — the exact `gfx` targets each PyTorch ROCm wheel is compiled for.
- [LLVM AMDGPU processors (gfx targets)](https://llvm.org/docs/AMDGPUUsage.html#processors)
  — maps each AMD GPU to its `gfxNNNN` ISA name.
- [ROCm changelog / release notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html)
  — when specific GPUs (e.g. `gfx900`, `gfx906`) were deprecated or removed.

## What the table answers

For every published `torch` wheel:

- which **CUDA** (and ROCm / CPU / XPU) build variants exist,
- which **Python** versions are supported,
- which **GPU compute capabilities** (SM architectures) are compiled in,
- for which **platforms** (linux / windows / macos, x86_64 / aarch64 / ...),
- the **release date**, whether it is on **PyPI**, and which CUDA build a bare `pip install torch`
  gives on Linux.

See [`data/COMPATIBILITY.md`](data/COMPATIBILITY.md) for the rendered table, plus
`data/compatibility_table.json` and `data/compatibility_matrix.csv`.

## Quick start (uv)

This is a [uv](https://docs.astral.sh/uv/) project.

```shell
# set up the environment (creates .venv from uv.lock)
uv sync

# generate everything: data/ table + docs/index.html (uses cache; ~seconds once warm)
uv run python generate.py

# first run / refresh: bypass cache and refetch all upstream sources (a few minutes)
uv run python generate.py --force

# regenerate only the table, skip the HTML
uv run python generate.py --no-html

# debug a single (torch, cuda) pair
uv run python generate.py --arch 2.8.0 12.8
#   -> capabilities: 7.0, 7.5, 8.0, 8.6, 9.0, 10.0, 12.0

# tests (hermetic, no network)
uv run pytest
```

The interactive picker has its own end-to-end test that loads `docs/index.html` in
[jsdom](https://github.com/jsdom/jsdom) (which executes the page's JavaScript) and drives the
selectors like a user, asserting the generated commands and UI state. It needs Node:

```shell
npm install     # once, to get jsdom
npm test        # runs test_html.mjs against docs/index.html
```

Regenerate `docs/index.html` (`uv run python generate.py`) before running it if you changed the
generator or the data.

No third-party **runtime** dependencies (standard library only); `pytest` is the sole dev
dependency. If you prefer not to use uv, plain `python generate.py` works too.

## Data sources (why it is deterministic)

| Fact | Source of truth | Format |
| --- | --- | --- |
| torch × backend × Python × platform | official wheel index `https://download.pytorch.org/whl/torch/` | wheel filenames (PEP 427/503) |
| release date per torch version | same wheel index | `data-upload-time` attribute (earliest wheel per version) |
| published to PyPI (vs alt registry) | PyPI JSON API `https://pypi.org/pypi/torch/json` | filename match (PyPI has no `+local` tags) |
| default `pip install` CUDA (Linux) | PyPI wheel METADATA (`.metadata` sidecar) | pinned `nvidia-*-cuXX` / `cuda-toolkit==A.B` dep |
| CUDA compute capabilities per (torch, CUDA) | PyTorch manywheel build scripts (`TORCH_CUDA_ARCH_LIST`) | shell `case` **or** Python dict |
| NVIDIA GPU → compute capability | curated from [developer.nvidia.com/cuda/gpus](https://developer.nvidia.com/cuda/gpus) (+ [legacy](https://developer.nvidia.com/cuda/gpus/legacy)) | `torch_compat/gpus.py` |
| AMD GPU → gfx target + ROCm support window | curated from the [AMD ROCm compatibility matrix](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) | `torch_compat/gpus.py` |

The compute-capability set is exactly what the build pipeline compiles into each wheel, so it is
read straight from the build script rather than guessed. Three upstream layouts are handled
automatically:

- `pytorch/builder` `@ release/<x.y>` → `manywheel/build_cuda.sh` (and `build.sh` for ≤ 1.11) — torch ≤ 2.5
- `pytorch/pytorch` `@ v<ver>` → `.ci/manywheel/build_cuda.sh` — torch 2.6 .. 2.12
- `pytorch/pytorch` `@ v<ver>` / `main` → `.ci/manywheel/build_env_setup.py` (`TORCH_CUDA_ARCH_LIST_TABLE`) — newest / nightly

The shell `case` evaluator faithfully reproduces the semantics we depend on: literal lists,
`${TORCH_CUDA_ARCH_LIST}` appends, glob labels (`11.[67]`, `10.*`), `$(...)` command
substitution, and the `aarch64` / `libtorch` `if` guards (which are false for the standard
x86_64 Python wheel, so the `else`/default branch is taken).

Every raw upstream artifact is cached under `data/raw/` (hits **and** 404s), so a full run is
auditable and repeatable, and reruns are fast and can work offline.

The GPU list is the one dataset that is not machine-scrapable from a single file, so it is curated
by hand in `torch_compat/gpus.py`. Compute capability is a stable hardware property — adding a new
card is a one-line entry.

## Extending to newer versions

Nothing to edit for the table — once PyTorch publishes new wheels and the matching build script,
run `uv run python generate.py --force` and the new rows (and the picker) update. If upstream
introduces a *new* build-script layout, add a URL to `candidate_urls()` and/or a small branch in
`evaluate()` in `torch_compat/arch_lists.py`, then add a snippet-based test to
`test_torch_compat.py`. For a brand-new GPU, add an entry to `torch_compat/gpus.py`.

## Layout

```
generate.py                 CLI orchestrator (table + HTML)
torch_compat/
  sources.py                HTTP fetch + on-disk (positive/negative) cache
  wheels.py                 parse the wheel index -> (torch, backend, python, platform)
  arch_lists.py             resolve build scripts + evaluate TORCH_CUDA_ARCH_LIST
  gpus.py                   curated NVIDIA GPU -> compute-capability dataset
  table.py                  combine + render JSON / CSV / Markdown
  html.py                   render the self-contained interactive picker
test_torch_compat.py        hermetic parser tests (pytest)
test_html.mjs               end-to-end jsdom test for docs/index.html (node)
package.json                dev dependency (jsdom) + `npm test` for the web test
data/                       generated table outputs (+ raw/ cache, gitignored)
docs/index.html             generated interactive picker (self-contained)
```

## Known limitations

- Compute capabilities for torch **< 1.9** are not recovered: those `pytorch/builder` release
  branches do not exist upstream. Such rows still list Python/platform info, with an empty
  capability set (shown as `?` in the Markdown).
- Nightly / `.dev` builds resolve against `main`, which is a moving target; released tags are
  fully reproducible. The picker hides `.dev` versions.
- Compute capabilities are reported for the standard x86_64 Python wheel. Some CUDA versions ship
  a slightly different arch list for `aarch64` or `libtorch` builds.
- The GPU lists cover common desktop/workstation/datacenter/Jetson cards; obscure OEM/mobile
  variants may be missing (add them to `gpus.py`).
- AMD ROCm support windows in `gpus.py` are curated from AMD's docs and are approximate at the
  patch level; the picker matches a card to a PyTorch ROCm wheel by ROCm major.minor version.
