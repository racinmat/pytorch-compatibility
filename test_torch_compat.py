"""Hermetic tests for the parsers (no network). Run: `python -m pytest` or `python test_torch_compat.py`.

Each snippet below is copied verbatim from a real PyTorch build script so the tests
lock in faithful evaluation of every upstream format we support.
"""

from __future__ import annotations

from torch_compat import arch_lists, gpus, html, table, wheels

# --- CUDA arch-list evaluation ---------------------------------------------

SHELL_1_13 = """
TORCH_CUDA_ARCH_LIST="3.7;5.0;6.0;7.0"
case ${CUDA_VERSION} in
    11.[3567])
        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST};7.5;8.0;8.6"
        ;;
    10.*)
        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
        ;;
    *)
        echo "unknown cuda version $CUDA_VERSION"
        exit 1
        ;;
esac
"""

SHELL_2_5_AARCH64 = """
TORCH_CUDA_ARCH_LIST="5.0;6.0;7.0;7.5;8.0;8.6"
case ${CUDA_VERSION} in
    12.4)
        if [[ "$GPU_ARCH_TYPE" = "cuda-aarch64" ]]; then
            TORCH_CUDA_ARCH_LIST="9.0"
        else
            TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST};9.0"
        fi
        ;;
    *) echo "unknown"; exit 1 ;;
esac
"""

SHELL_2_10_INLINE = """
TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
case ${CUDA_VERSION} in
    12.6) TORCH_CUDA_ARCH_LIST="5.0;6.0;${TORCH_CUDA_ARCH_LIST}" ;;
    12.8) TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST};10.0;12.0" ;;
    13.0)
        TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;$([[ "$ARCH" == "aarch64" ]] && echo "11.0;" || echo "")12.0+PTX"
        ;;
    *) echo "unknown cuda version $CUDA_VERSION"; exit 1 ;;
esac
"""

PY_TABLE = """
TORCH_CUDA_ARCH_LIST_TABLE: dict[str, dict[str, set[int]]] = {
    "12.6": {"x86_64": {50, 60, 70, 75, 80, 86, 90}, "aarch64": {80, 90}},
    "13.0": {"x86_64": {75, 80, 86, 90, 100, 120}, "aarch64": {80, 90, 100, 110, 120}},
}
"""


def test_shell_glob_and_append():
    assert arch_lists.evaluate(SHELL_1_13, "11.6") == "3.7;5.0;6.0;7.0;7.5;8.0;8.6"
    assert arch_lists.evaluate(SHELL_1_13, "10.2") == "3.7;5.0;6.0;7.0"
    assert arch_lists.evaluate(SHELL_1_13, "12.1") is None  # only *) matches -> unsupported


def test_shell_aarch64_takes_else_branch():
    # default x86_64 wheel: guard false -> else branch appends to the base list
    assert arch_lists.evaluate(SHELL_2_5_AARCH64, "12.4") == "5.0;6.0;7.0;7.5;8.0;8.6;9.0"


def test_shell_inline_and_command_substitution():
    assert arch_lists.evaluate(SHELL_2_10_INLINE, "12.6") == "5.0;6.0;7.0;7.5;8.0;8.6;9.0"
    assert arch_lists.evaluate(SHELL_2_10_INLINE, "12.8") == "7.0;7.5;8.0;8.6;9.0;10.0;12.0"
    # `$([[ "$ARCH" == aarch64 ]] && ... || echo "")` -> empty for the x86_64 wheel
    assert arch_lists.evaluate(SHELL_2_10_INLINE, "13.0") == "7.5;8.0;8.6;9.0;10.0;12.0+PTX"


def test_python_table_form():
    assert arch_lists.evaluate(PY_TABLE, "12.6") == "5.0;6.0;7.0;7.5;8.0;8.6;9.0"
    assert arch_lists.evaluate(PY_TABLE, "13.0") == "7.5;8.0;8.6;9.0;10.0;12.0"
    assert arch_lists.evaluate(PY_TABLE, "99.9") is None


def test_capabilities_and_ptx():
    caps, ptx = arch_lists._to_capabilities("7.5;8.0;9.0;10.0;12.0+PTX")
    assert caps == ("7.5", "8.0", "9.0", "10.0", "12.0")
    assert ptx is True


# --- wheel filename parsing -------------------------------------------------

def test_wheel_parse_cuda():
    (row,) = wheels.parse_index('<a>torch-2.3.0+cu121-cp311-cp311-linux_x86_64.whl</a>')
    assert (row.torch_version, row.backend, row.backend_version) == ("2.3.0", "cuda", "12.1")
    assert (row.python, row.os, row.arch) == ("3.11", "linux", "x86_64")


def test_wheel_parse_build_tag_not_mistaken_for_python():
    (row,) = wheels.parse_index('<a>torch-2.0.0-1-cp38-cp38-manylinux1_x86_64.whl</a>')
    assert row.python == "3.8"  # the "-1-" build tag must not be read as python "1"


def test_wheel_parse_upload_date():
    html = (
        '<a href="x/torch-2.3.0+cu121-cp311-cp311-linux_x86_64.whl#sha256=ab"'
        ' data-upload-time="2024-04-24T18:12:00Z">'
        'torch-2.3.0+cu121-cp311-cp311-linux_x86_64.whl</a>'
    )
    (row,) = wheels.parse_index(html)
    assert row.upload_date == "2024-04-24"


def test_wheel_parse_variants():
    html = (
        '<a>torch-2.5.0+cpu-cp312-cp312-win_amd64.whl</a>'
        '<a>torch-2.5.0+rocm6.2-cp310-cp310-linux_x86_64.whl</a>'
        '<a>torch-2.3.0-cp311-none-macosx_11_0_arm64.whl</a>'
    )
    rows = {(r.backend, r.backend_version, r.os, r.arch) for r in wheels.parse_index(html)}
    assert ("cpu", "", "windows", "x86_64") in rows
    assert ("rocm", "6.2", "linux", "x86_64") in rows
    assert ("default", "", "macos", "arm64") in rows


def test_pypi_filenames_and_flags():
    payload = (
        '{"releases": {'
        '"2.3.0": [{"filename": "torch-2.3.0-cp311-cp311-manylinux1_x86_64.whl"}],'
        '"2.3.0.dev1": [{"filename": "torch-2.3.0.dev1-cp311-cp311-linux_x86_64.whl"}]'
        '}}'
    )
    pypi = wheels.parse_pypi_filenames(payload)
    assert "torch-2.3.0-cp311-cp311-manylinux1_x86_64.whl" in pypi

    html = (
        # untagged default build -> matches a PyPI filename
        '<a>torch-2.3.0-cp311-cp311-manylinux1_x86_64.whl</a>'
        # +cu121 tagged build -> never on PyPI
        '<a>torch-2.3.0+cu121-cp311-cp311-linux_x86_64.whl</a>'
    )
    rows = table.build_rows(wheels.parse_index(html), pypi_filenames=pypi)
    by_backend = {r.backend: r for r in rows}
    assert by_backend["default"].on_pypi is True
    assert by_backend["cuda"].on_pypi is False
    # version-level flag is true for both rows because the version exists on PyPI
    assert by_backend["cuda"].version_on_pypi is True


def test_parse_default_cuda_both_metadata_formats():
    new = 'Requires-Dist: cuda-toolkit[cudart,nvrtc]==13.0.3; platform_system == "Linux"'
    mid = 'Requires-Dist: nvidia-cuda-runtime-cu12==12.6.77; platform_system == "Linux"'
    old = 'Requires-Dist: nvidia-cuda-runtime-cu12 (==12.4.127) ; platform_system == "Linux"'
    assert wheels.parse_default_cuda(new) == "13.0"
    assert wheels.parse_default_cuda(mid) == "12.6"
    assert wheels.parse_default_cuda(old) == "12.4"
    assert wheels.parse_default_cuda("no cuda deps here") == ""


def test_pypi_linux_default_marks_single_cuda_row():
    html = (
        '<a>torch-2.13.0+cu126-cp311-cp311-linux_x86_64.whl</a>'
        '<a>torch-2.13.0+cu130-cp311-cp311-linux_x86_64.whl</a>'
        '<a>torch-2.13.0+cu132-cp311-cp311-linux_x86_64.whl</a>'
    )
    rows = table.build_rows(
        wheels.parse_index(html), pypi_default_cuda={"2.13.0": "13.0"}
    )
    marked = {r.backend_version: r.pypi_linux_default for r in rows if r.backend == "cuda"}
    assert marked == {"12.6": False, "13.0": True, "13.2": False}


# --- interactive HTML payload ----------------------------------------------

def test_html_channel_and_label():
    assert html._channel("cuda", "12.9") == "cu129"
    assert html._channel("cuda", "13.0") == "cu130"
    assert html._channel("rocm", "6.2") == "rocm6.2"
    assert html._channel("cpu", "") == "cpu"
    assert html._channel("xpu", "") == "xpu"
    assert html._channel("default", "") is None
    assert html._label("cuda", "12.1") == "CUDA 12.1"
    assert html._label("rocm", "6.2") == "ROCm 6.2"


def test_html_payload_shape_and_dev_filtering():
    src = (
        '<a>torch-2.5.0+cu124-cp311-cp311-linux_x86_64.whl</a>'
        '<a>torch-2.5.0+cpu-cp311-cp311-win_amd64.whl</a>'
        '<a>torch-2.6.0.dev1+cu124-cp311-cp311-linux_x86_64.whl</a>'
    )
    rows = table.build_rows(wheels.parse_index(src))
    payload = html.build_payload(rows)
    versions = {v["version"] for v in payload["versions"]}
    assert "2.5.0" in versions
    assert "2.6.0.dev1" not in versions  # dev releases are hidden from the picker

    v = next(v for v in payload["versions"] if v["version"] == "2.5.0")
    cuda = next(b for b in v["builds"] if b["backend"] == "cuda")
    assert cuda["channel"] == "cu124"
    assert cuda["os"] == ["linux"]
    cpu = next(b for b in v["builds"] if b["backend"] == "cpu")
    assert cpu["channel"] == "cpu" and cpu["os"] == ["windows"]


def test_gpu_dataset_valid():
    assert gpus.GPUS, "GPU list must not be empty"
    names = [g.name for g in gpus.GPUS]
    assert len(names) == len(set(names)), "duplicate GPU names"
    for g in gpus.GPUS:
        assert g.vendor in ("nvidia", "amd")
        if g.vendor == "nvidia":
            major, _, minor = g.cc.partition(".")
            assert major.isdigit() and minor.isdigit(), f"bad cc for {g.name}: {g.cc}"
            assert not g.gfx
        else:
            assert g.gfx.startswith("gfx"), f"bad gfx for {g.name}: {g.gfx}"
            assert not g.cc
            maj, _, minr = g.rocm_min.partition(".")
            assert maj.isdigit() and minr.isdigit(), f"bad rocm_min for {g.name}: {g.rocm_min}"
    by_name = {g.name: g for g in gpus.GPUS}
    assert by_name["GeForce RTX 4090"].cc == "8.9"
    assert by_name["A100"].cc == "8.0"
    assert by_name["H100"].cc == "9.0"
    assert by_name["Instinct MI300X"].gfx == "gfx942"
    assert by_name["Instinct MI300X"].rocm_min == "6.0"
    assert by_name["Radeon RX 7900 XTX"].vendor == "amd"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
