"""Generate the PyTorch / CUDA / compute-capability / Python compatibility table.

All data is extracted deterministically from upstream metadata and source code:

    * torch x CUDA/ROCm/CPU x Python x platform  -> the official wheel index
      (https://download.pytorch.org/whl/torch/)
    * CUDA compute capabilities per (torch, CUDA)  -> the PyTorch manywheel build
      scripts (TORCH_CUDA_ARCH_LIST in build_cuda.sh)

Raw inputs are cached under data/raw/ so runs are reproducible and auditable.
Adding support for a newer torch/CUDA release requires no code changes: rerun with
--force once the upstream wheel index and build scripts are published.

Usage:
    python generate.py                 # generate into ./data
    python generate.py --force         # bypass cache, refetch everything
    python generate.py --arch 2.4 12.1 # print one arch list and exit (debug)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch_compat import arch_lists, html, table

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data"
HTML_OUT = ROOT / "docs" / "index.html"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="Ignore cache and refetch upstream sources.")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="Output directory (default: ./data).")
    parser.add_argument(
        "--html-out",
        type=Path,
        default=HTML_OUT,
        help="Interactive HTML picker output path (default: ./docs/index.html).",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip generating the interactive HTML picker.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    parser.add_argument(
        "--arch",
        nargs=2,
        metavar=("TORCH", "CUDA"),
        help="Debug: print the TORCH_CUDA_ARCH_LIST for one (torch, cuda) pair and exit.",
    )
    args = parser.parse_args(argv)

    if args.arch:
        torch_v, cuda_v = args.arch
        res = arch_lists.arch_result(torch_v, cuda_v, force=args.force)
        if res is None:
            print(f"No arch list found for torch {torch_v} + CUDA {cuda_v}", file=sys.stderr)
            return 1
        print(f"torch {torch_v} + CUDA {cuda_v}")
        print(f"  raw:          {res.arch_list_raw}")
        print(f"  capabilities: {', '.join(res.capabilities)}")
        print(f"  +PTX:         {res.has_ptx}")
        print(f"  source:       {res.source_url}")
        return 0

    rows = table.generate(args.out, force=args.force, verbose=not args.quiet)
    cuda_rows = sum(1 for r in rows if r.backend == "cuda")
    if not args.no_html:
        html.write_html(rows, args.html_out)
    if not args.quiet:
        print(f"\nDone. {len(rows)} rows ({cuda_rows} CUDA) written to {args.out}")
        for name in ("compatibility_table.json", "compatibility_matrix.csv", "COMPATIBILITY.md"):
            print(f"  - {args.out / name}")
        if not args.no_html:
            print(f"  - {args.html_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
