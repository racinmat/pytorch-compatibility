"""Deterministic extraction of PyTorch compatibility data.

Sub-modules:
    sources     - HTTP fetching with on-disk caching of raw upstream artifacts.
    wheels      - Parse the official wheel index into (torch, backend, python, platform) rows.
    arch_lists  - Evaluate ``TORCH_CUDA_ARCH_LIST`` from PyTorch build scripts (source of truth
                  for supported CUDA compute capabilities per CUDA version).
    gpus        - Curated NVIDIA GPU -> compute-capability dataset.
    table       - Combine the sources into a compatibility table (JSON / CSV / Markdown).
    html        - Render the self-contained interactive install-command picker.
"""

__all__ = ["sources", "wheels", "arch_lists", "gpus", "table", "html"]
