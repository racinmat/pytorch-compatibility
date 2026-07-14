"""Curated NVIDIA GPU -> CUDA compute capability (SM) dataset.

Compute capabilities are a stable hardware property published by NVIDIA
(https://developer.nvidia.com/cuda/gpus). Unlike the rest of this project the
list cannot be scraped from a single machine-readable file, so it is maintained
here by hand. Each entry maps a marketing name to its compute capability and the
GPU microarchitecture / market segment (used only for grouping in the UI).

Add new cards as they ship; the value that matters for wheel compatibility is
``cc`` (``"major.minor"``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Gpu:
    name: str
    cc: str = ""       # NVIDIA compute capability "major.minor" ("" for AMD)
    arch: str = ""     # microarchitecture (Ampere, Ada Lovelace, RDNA3, CDNA3, ...)
    segment: str = ""  # consumer | workstation | datacenter | jetson
    vendor: str = "nvidia"
    gfx: str = ""      # AMD LLVM target, e.g. "gfx1100" ("" for NVIDIA)
    rocm_min: str = "" # AMD: earliest ROCm release with support ("major.minor")
    rocm_max: str = "" # AMD: last ROCm release with support ("" = still current)


# Documentation sources for the compute-capability / gfx target of each vendor.
SOURCES = {
    "nvidia": "https://developer.nvidia.com/cuda/gpus",
    "nvidia_legacy": "https://developer.nvidia.com/cuda/gpus/legacy",
    "amd_gpus": "https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html",
    "amd_gfx": "https://llvm.org/docs/AMDGPUUsage.html#processors",
}


def _amd(name: str, gfx: str, arch: str, segment: str, rocm_min: str, rocm_max: str = "") -> "Gpu":
    return Gpu(
        name=name, arch=arch, segment=segment, vendor="amd",
        gfx=gfx, rocm_min=rocm_min, rocm_max=rocm_max,
    )


# Ordered newest-architecture-first so the UI groups read naturally.
GPUS: list[Gpu] = [
    # --- Blackwell (consumer / workstation, sm_120) -------------------------
    Gpu("GeForce RTX 5090", "12.0", "Blackwell", "consumer"),
    Gpu("GeForce RTX 5080", "12.0", "Blackwell", "consumer"),
    Gpu("GeForce RTX 5070 Ti", "12.0", "Blackwell", "consumer"),
    Gpu("GeForce RTX 5070", "12.0", "Blackwell", "consumer"),
    Gpu("GeForce RTX 5060 Ti", "12.0", "Blackwell", "consumer"),
    Gpu("GeForce RTX 5060", "12.0", "Blackwell", "consumer"),
    Gpu("RTX PRO 6000 Blackwell", "12.0", "Blackwell", "workstation"),
    Gpu("RTX PRO 5000 Blackwell", "12.0", "Blackwell", "workstation"),
    Gpu("RTX PRO 4500 Blackwell", "12.0", "Blackwell", "workstation"),
    # --- Blackwell (datacenter, sm_100) ------------------------------------
    Gpu("B200", "10.0", "Blackwell", "datacenter"),
    Gpu("B100", "10.0", "Blackwell", "datacenter"),
    Gpu("GB200 (Grace Blackwell)", "10.0", "Blackwell", "datacenter"),
    # --- Hopper (sm_90) ----------------------------------------------------
    Gpu("H200", "9.0", "Hopper", "datacenter"),
    Gpu("H100", "9.0", "Hopper", "datacenter"),
    Gpu("H800", "9.0", "Hopper", "datacenter"),
    Gpu("GH200 (Grace Hopper)", "9.0", "Hopper", "datacenter"),
    # --- Ada Lovelace (sm_89) ----------------------------------------------
    Gpu("GeForce RTX 4090", "8.9", "Ada Lovelace", "consumer"),
    Gpu("GeForce RTX 4080 SUPER", "8.9", "Ada Lovelace", "consumer"),
    Gpu("GeForce RTX 4080", "8.9", "Ada Lovelace", "consumer"),
    Gpu("GeForce RTX 4070 Ti SUPER", "8.9", "Ada Lovelace", "consumer"),
    Gpu("GeForce RTX 4070 Ti", "8.9", "Ada Lovelace", "consumer"),
    Gpu("GeForce RTX 4070 SUPER", "8.9", "Ada Lovelace", "consumer"),
    Gpu("GeForce RTX 4070", "8.9", "Ada Lovelace", "consumer"),
    Gpu("GeForce RTX 4060 Ti", "8.9", "Ada Lovelace", "consumer"),
    Gpu("GeForce RTX 4060", "8.9", "Ada Lovelace", "consumer"),
    Gpu("RTX 6000 Ada", "8.9", "Ada Lovelace", "workstation"),
    Gpu("RTX 5000 Ada", "8.9", "Ada Lovelace", "workstation"),
    Gpu("RTX 4500 Ada", "8.9", "Ada Lovelace", "workstation"),
    Gpu("RTX 4000 Ada", "8.9", "Ada Lovelace", "workstation"),
    Gpu("L40S", "8.9", "Ada Lovelace", "datacenter"),
    Gpu("L40", "8.9", "Ada Lovelace", "datacenter"),
    Gpu("L4", "8.9", "Ada Lovelace", "datacenter"),
    # --- Ampere (sm_86 / sm_80 / sm_87) ------------------------------------
    Gpu("GeForce RTX 3090 Ti", "8.6", "Ampere", "consumer"),
    Gpu("GeForce RTX 3090", "8.6", "Ampere", "consumer"),
    Gpu("GeForce RTX 3080 Ti", "8.6", "Ampere", "consumer"),
    Gpu("GeForce RTX 3080", "8.6", "Ampere", "consumer"),
    Gpu("GeForce RTX 3070 Ti", "8.6", "Ampere", "consumer"),
    Gpu("GeForce RTX 3070", "8.6", "Ampere", "consumer"),
    Gpu("GeForce RTX 3060 Ti", "8.6", "Ampere", "consumer"),
    Gpu("GeForce RTX 3060", "8.6", "Ampere", "consumer"),
    Gpu("GeForce RTX 3050", "8.6", "Ampere", "consumer"),
    Gpu("RTX A6000", "8.6", "Ampere", "workstation"),
    Gpu("RTX A5000", "8.6", "Ampere", "workstation"),
    Gpu("RTX A4000", "8.6", "Ampere", "workstation"),
    Gpu("RTX A2000", "8.6", "Ampere", "workstation"),
    Gpu("A40", "8.6", "Ampere", "datacenter"),
    Gpu("A10", "8.6", "Ampere", "datacenter"),
    Gpu("A16", "8.6", "Ampere", "datacenter"),
    Gpu("A2", "8.6", "Ampere", "datacenter"),
    Gpu("A100", "8.0", "Ampere", "datacenter"),
    Gpu("A30", "8.0", "Ampere", "datacenter"),
    Gpu("Jetson AGX Orin", "8.7", "Ampere", "jetson"),
    Gpu("Jetson Orin NX", "8.7", "Ampere", "jetson"),
    Gpu("Jetson Orin Nano", "8.7", "Ampere", "jetson"),
    # --- Turing (sm_75) ----------------------------------------------------
    Gpu("Titan RTX", "7.5", "Turing", "consumer"),
    Gpu("GeForce RTX 2080 Ti", "7.5", "Turing", "consumer"),
    Gpu("GeForce RTX 2080 SUPER", "7.5", "Turing", "consumer"),
    Gpu("GeForce RTX 2080", "7.5", "Turing", "consumer"),
    Gpu("GeForce RTX 2070 SUPER", "7.5", "Turing", "consumer"),
    Gpu("GeForce RTX 2070", "7.5", "Turing", "consumer"),
    Gpu("GeForce RTX 2060 SUPER", "7.5", "Turing", "consumer"),
    Gpu("GeForce RTX 2060", "7.5", "Turing", "consumer"),
    Gpu("GeForce GTX 1660 Ti", "7.5", "Turing", "consumer"),
    Gpu("GeForce GTX 1660 SUPER", "7.5", "Turing", "consumer"),
    Gpu("GeForce GTX 1660", "7.5", "Turing", "consumer"),
    Gpu("GeForce GTX 1650 SUPER", "7.5", "Turing", "consumer"),
    Gpu("GeForce GTX 1650", "7.5", "Turing", "consumer"),
    Gpu("Quadro RTX 8000", "7.5", "Turing", "workstation"),
    Gpu("Quadro RTX 6000", "7.5", "Turing", "workstation"),
    Gpu("Quadro RTX 5000", "7.5", "Turing", "workstation"),
    Gpu("Tesla T4", "7.5", "Turing", "datacenter"),
    # --- Volta (sm_70 / sm_72) ---------------------------------------------
    Gpu("Titan V", "7.0", "Volta", "consumer"),
    Gpu("Tesla V100", "7.0", "Volta", "datacenter"),
    Gpu("Quadro GV100", "7.0", "Volta", "workstation"),
    Gpu("Jetson AGX Xavier", "7.2", "Volta", "jetson"),
    Gpu("Jetson Xavier NX", "7.2", "Volta", "jetson"),
    # --- Pascal (sm_60 / sm_61 / sm_62) ------------------------------------
    Gpu("Titan Xp", "6.1", "Pascal", "consumer"),
    Gpu("Titan X (Pascal)", "6.1", "Pascal", "consumer"),
    Gpu("GeForce GTX 1080 Ti", "6.1", "Pascal", "consumer"),
    Gpu("GeForce GTX 1080", "6.1", "Pascal", "consumer"),
    Gpu("GeForce GTX 1070 Ti", "6.1", "Pascal", "consumer"),
    Gpu("GeForce GTX 1070", "6.1", "Pascal", "consumer"),
    Gpu("GeForce GTX 1060", "6.1", "Pascal", "consumer"),
    Gpu("GeForce GTX 1050 Ti", "6.1", "Pascal", "consumer"),
    Gpu("GeForce GTX 1050", "6.1", "Pascal", "consumer"),
    Gpu("GeForce GT 1030", "6.1", "Pascal", "consumer"),
    Gpu("Tesla P40", "6.1", "Pascal", "datacenter"),
    Gpu("Tesla P4", "6.1", "Pascal", "datacenter"),
    Gpu("Quadro P6000", "6.1", "Pascal", "workstation"),
    Gpu("Quadro P5000", "6.1", "Pascal", "workstation"),
    Gpu("Tesla P100", "6.0", "Pascal", "datacenter"),
    Gpu("Quadro GP100", "6.0", "Pascal", "workstation"),
    Gpu("Jetson TX2", "6.2", "Pascal", "jetson"),
    # --- Maxwell (sm_50 / sm_52 / sm_53) -----------------------------------
    Gpu("GeForce GTX Titan X (Maxwell)", "5.2", "Maxwell", "consumer"),
    Gpu("GeForce GTX 980 Ti", "5.2", "Maxwell", "consumer"),
    Gpu("GeForce GTX 980", "5.2", "Maxwell", "consumer"),
    Gpu("GeForce GTX 970", "5.2", "Maxwell", "consumer"),
    Gpu("GeForce GTX 960", "5.2", "Maxwell", "consumer"),
    Gpu("GeForce GTX 950", "5.2", "Maxwell", "consumer"),
    Gpu("Quadro M6000", "5.2", "Maxwell", "workstation"),
    Gpu("Tesla M40", "5.2", "Maxwell", "datacenter"),
    Gpu("Tesla M60", "5.2", "Maxwell", "datacenter"),
    Gpu("GeForce GTX 750 Ti", "5.0", "Maxwell", "consumer"),
    Gpu("GeForce GTX 750", "5.0", "Maxwell", "consumer"),
    Gpu("Jetson Nano", "5.3", "Maxwell", "jetson"),
    Gpu("Jetson TX1", "5.3", "Maxwell", "jetson"),
    # --- Kepler (sm_35 / sm_37) -- dropped by modern wheels, kept for context
    Gpu("GeForce GTX Titan Z", "3.5", "Kepler", "consumer"),
    Gpu("GeForce GTX Titan Black", "3.5", "Kepler", "consumer"),
    Gpu("GeForce GTX 780 Ti", "3.5", "Kepler", "consumer"),
    Gpu("GeForce GTX 780", "3.5", "Kepler", "consumer"),
    Gpu("Tesla K40", "3.5", "Kepler", "datacenter"),
    Gpu("Tesla K80", "3.7", "Kepler", "datacenter"),
]

# --- AMD -------------------------------------------------------------------
# ROCm support windows (rocm_min .. rocm_max) are curated from AMD's ROCm
# "compatibility matrix" / system-requirements docs (SOURCES["amd_gpus"]); the
# LLVM gfx target is the AMD GPU ISA name (SOURCES["amd_gfx"]). A PyTorch ROCm
# wheel of version V works on a card when rocm_min <= V <= (rocm_max or now).
#
# Only GPUs that PyTorch's prebuilt ROCm wheels actually target are listed. A
# wheel is compiled for a fixed set of gfx ISAs (PYTORCH_ROCM_ARCH) and ROCm's
# HIP runtime + math libraries (rocBLAS/MIOpen) only ship kernels for those, so
# a card with no matching gfx binary simply has no usable wheel. Deliberately
# NOT listed, because they are not supported compute targets:
#   * RDNA1  - Radeon RX 5000 series (gfx1010/1011/1012, e.g. RX 5700 / 5700 XT
#              / 5600 / 5500): never an official ROCm compute target; no wheel.
#              (The HSA_OVERRIDE_GFX_VERSION=10.3.0 "pretend to be gfx1030" hack
#              is unofficial, unsupported and unreliable, so it is excluded.)
#   * Polaris/older GCN - RX 400/500 (gfx803) and earlier: only ever had
#              experimental support in very old ROCm; dropped long ago.
# Oldest cards that DO work: gfx900 (RX Vega 56/64) on ROCm <= 5.4, gfx906
# (Radeon VII / Instinct MI50/MI60) on ROCm <= 5.7; on current ROCm 6.x the
# oldest usable are gfx908 (Instinct MI100) and gfx1030 (RDNA2, RX 6800+).
GPUS += [
    # --- RDNA4 (gfx12) -----------------------------------------------------
    _amd("Radeon RX 9070 XT", "gfx1201", "RDNA4", "consumer", "6.4"),
    _amd("Radeon RX 9070", "gfx1201", "RDNA4", "consumer", "6.4"),
    # --- CDNA3 (Instinct, gfx942) -----------------------------------------
    _amd("Instinct MI325X", "gfx942", "CDNA3", "datacenter", "6.2"),
    _amd("Instinct MI300X", "gfx942", "CDNA3", "datacenter", "6.0"),
    _amd("Instinct MI300A", "gfx942", "CDNA3", "datacenter", "6.0"),
    # --- CDNA2 (Instinct, gfx90a) -----------------------------------------
    _amd("Instinct MI250X", "gfx90a", "CDNA2", "datacenter", "5.0"),
    _amd("Instinct MI250", "gfx90a", "CDNA2", "datacenter", "5.0"),
    _amd("Instinct MI210", "gfx90a", "CDNA2", "datacenter", "5.0"),
    # --- CDNA1 (Instinct, gfx908) -----------------------------------------
    _amd("Instinct MI100", "gfx908", "CDNA", "datacenter", "4.0"),
    # --- RDNA3 (gfx11) -----------------------------------------------------
    _amd("Radeon PRO W7900", "gfx1100", "RDNA3", "workstation", "5.7"),
    _amd("Radeon PRO W7800", "gfx1100", "RDNA3", "workstation", "5.7"),
    _amd("Radeon RX 7900 XTX", "gfx1100", "RDNA3", "consumer", "5.7"),
    _amd("Radeon RX 7900 XT", "gfx1100", "RDNA3", "consumer", "5.7"),
    _amd("Radeon RX 7900 GRE", "gfx1100", "RDNA3", "consumer", "6.0"),
    _amd("Radeon RX 7800 XT", "gfx1101", "RDNA3", "consumer", "6.1"),
    _amd("Radeon RX 7700 XT", "gfx1101", "RDNA3", "consumer", "6.1"),
    # --- RDNA2 (gfx1030) ---------------------------------------------------
    _amd("Radeon PRO W6800", "gfx1030", "RDNA2", "workstation", "5.0"),
    _amd("Radeon RX 6950 XT", "gfx1030", "RDNA2", "consumer", "5.2"),
    _amd("Radeon RX 6900 XT", "gfx1030", "RDNA2", "consumer", "5.2"),
    _amd("Radeon RX 6800 XT", "gfx1030", "RDNA2", "consumer", "5.2"),
    _amd("Radeon RX 6800", "gfx1030", "RDNA2", "consumer", "5.2"),
    # --- GCN5 / Vega (gfx900 / gfx906) -- dropped by recent ROCm -----------
    _amd("Radeon PRO VII", "gfx906", "Vega 20", "workstation", "3.0", "5.7"),
    _amd("Radeon VII", "gfx906", "Vega 20", "consumer", "2.9", "5.7"),
    _amd("Instinct MI50", "gfx906", "Vega 20", "datacenter", "3.0", "5.7"),
    _amd("Instinct MI60", "gfx906", "Vega 20", "datacenter", "3.0", "5.7"),
    _amd("Radeon RX Vega 64", "gfx900", "Vega 10", "consumer", "2.0", "5.4"),
    _amd("Radeon RX Vega 56", "gfx900", "Vega 10", "consumer", "2.0", "5.4"),
]
