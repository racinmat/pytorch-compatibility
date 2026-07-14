"""Render a self-contained interactive install-command picker (docs/index.html).

The page embeds the generated compatibility data and the curated GPU list as
JSON, so it works when opened directly from disk (``file://``) with no server
and no network. Given a Python version, GPU / compute platform, torch version
and package manager it prints ready-to-paste install commands (including the
extra index/source config Poetry, PDM and uv projects need).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from . import gpus
from .table import CompatRow, version_key


def _channel(backend: str, backend_version: str) -> str | None:
    """The ``download.pytorch.org/whl/<channel>`` path segment, or None for PyPI."""
    if backend == "cuda":
        return "cu" + backend_version.replace(".", "")
    if backend == "rocm":
        return "rocm" + backend_version
    if backend == "cpu":
        return "cpu"
    if backend == "xpu":
        return "xpu"
    return None  # "default" -> plain PyPI wheel


def _label(backend: str, backend_version: str) -> str:
    return {
        "cuda": f"CUDA {backend_version}",
        "rocm": f"ROCm {backend_version}",
        "cpu": "CPU",
        "xpu": "XPU (Intel)",
        "default": "Default (PyPI)",
    }.get(backend, backend)


def _sort_rank(backend: str) -> int:
    return {"cuda": 0, "rocm": 1, "xpu": 2, "cpu": 3, "default": 4}.get(backend, 5)


def build_payload(rows: list[CompatRow]) -> dict:
    versions: dict[str, dict] = {}
    for r in rows:
        if "dev" in r.torch_version.lower():
            continue
        v = versions.setdefault(
            r.torch_version,
            {
                "version": r.torch_version,
                "released": r.release_date,
                "on_pypi": r.version_on_pypi,
                "builds": [],
            },
        )
        oses = sorted({p.split("/")[0] for p in r.platforms})
        v["builds"].append(
            {
                "key": r.backend + ("-" + r.backend_version if r.backend_version else ""),
                "backend": r.backend,
                "backend_version": r.backend_version,
                "label": _label(r.backend, r.backend_version),
                "channel": _channel(r.backend, r.backend_version),
                "os": oses,
                "pythons": r.pythons,
                "caps": r.compute_capabilities,
                "ptx": r.has_ptx,
                "pypi_default": r.pypi_linux_default,
                "on_pypi": r.on_pypi,
                "rank": _sort_rank(r.backend),
            }
        )

    for v in versions.values():
        v["builds"].sort(
            key=lambda b: (
                b["rank"],
                tuple(-x for x in version_key(b["backend_version"] or "0")),
            )
        )

    ordered = sorted(versions.values(), key=lambda v: version_key(v["version"]), reverse=True)
    return {"versions": ordered}


def write_html(rows: list[CompatRow], path: Path) -> None:
    payload = build_payload(rows)
    gpu_list = [asdict(g) for g in gpus.GPUS]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = (
        _TEMPLATE.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
        .replace("__GPUS__", json.dumps(gpu_list, separators=(",", ":")))
        .replace("__GENERATED__", generated)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PyTorch install command picker</title>
<style>
  :root {
    --orange: #ee4c2c;
    --ink: #1a1a1a;
    --muted: #6b7280;
    --line: #e5e7eb;
    --bg: #f7f7f8;
    --card: #ffffff;
    --ok: #157f3b;
    --warn: #b45309;
    --bad: #b91c1c;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  header { background: linear-gradient(135deg, #2d2d2d, #1a1a1a); color: #fff; padding: 28px 20px; }
  header .wrap { max-width: 1040px; margin: 0 auto; }
  header h1 { margin: 0 0 6px; font-size: 24px; }
  header h1 span { color: var(--orange); }
  header p { margin: 0; color: #cfcfcf; font-size: 14px; }
  main { max-width: 1040px; margin: 0 auto; padding: 20px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 18px; margin-bottom: 18px; }
  .row { display: grid; grid-template-columns: 150px 1fr; gap: 14px; align-items: start;
         padding: 12px 0; border-bottom: 1px solid var(--line); }
  .row:last-child { border-bottom: none; }
  .row > label { font-weight: 600; padding-top: 6px; }
  .opts { display: flex; flex-wrap: wrap; gap: 8px; }
  .opt {
    border: 1px solid var(--line); background: #fff; border-radius: 8px;
    padding: 7px 13px; cursor: pointer; font-size: 14px; user-select: none;
    display: inline-flex; align-items: center; gap: 7px; transition: all .12s;
  }
  .opt:hover { border-color: var(--orange); }
  .opt.active { background: var(--orange); border-color: var(--orange); color: #fff; }
  .badge { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .03em;
           padding: 1px 5px; border-radius: 4px; }
  .badge.ok { background: #e6f4ea; color: var(--ok); }
  .badge.ptx { background: #fef3c7; color: var(--warn); }
  .opt.active .badge { background: rgba(255,255,255,.85); }
  select { font-size: 14px; padding: 7px 10px; border: 1px solid var(--line);
           border-radius: 8px; background: #fff; min-width: 300px; max-width: 100%; }
  .combo { position: relative; max-width: 380px; }
  .combo input { width: 100%; font-size: 14px; padding: 8px 34px 8px 10px; border: 1px solid var(--line);
                 border-radius: 8px; background: #fff; }
  .combo input:focus { outline: none; border-color: var(--orange); }
  .combo .clear { position: absolute; top: 50%; right: 8px; transform: translateY(-50%);
                  border: none; background: transparent; color: var(--muted); font-size: 18px;
                  line-height: 1; cursor: pointer; padding: 0 4px; }
  .combo .clear:hover { color: var(--ink); }
  .combo-list { position: absolute; z-index: 20; left: 0; right: 0; margin-top: 4px; background: #fff;
                border: 1px solid var(--line); border-radius: 8px; box-shadow: 0 8px 28px rgba(0,0,0,.14);
                max-height: 300px; overflow-y: auto; display: none; }
  .combo-item { padding: 8px 11px; cursor: pointer; display: flex; justify-content: space-between;
                gap: 12px; align-items: center; }
  .combo-item:hover, .combo-item.hi { background: #f3f4f6; }
  .combo-item.sel { background: #fff3f0; font-weight: 600; }
  .combo-group { padding: 6px 11px 3px; font-size: 11px; text-transform: uppercase; letter-spacing: .04em;
                 color: var(--muted); background: #fafafa; position: sticky; top: 0; }
  .ci-meta { color: var(--muted); font-size: 12px; white-space: nowrap; }
  .checks { display: flex; flex-wrap: wrap; gap: 16px; padding-top: 6px; }
  .checks label { font-weight: 400; display: inline-flex; align-items: center; gap: 6px; cursor: pointer; }
  .gpu-info { margin-top: 10px; font-size: 14px; color: var(--muted); }
  .gpu-info b { color: var(--ink); }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; }
  .pill.ok { background: #e6f4ea; color: var(--ok); }
  .pill.ptx { background: #fef3c7; color: var(--warn); }
  .pill.no { background: #fee2e2; color: var(--bad); }
  .note-range { margin-top: 8px; font-size: 13px; color: var(--muted);
                background: #fafafa; border: 1px solid var(--line); border-radius: 8px; padding: 9px 11px; }
  .note-range b { color: var(--ink); }
  h2 { font-size: 15px; margin: 0 0 12px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
  .cmd { margin-bottom: 14px; }
  .cmd h3 { margin: 0 0 6px; font-size: 13px; color: var(--muted); font-weight: 600; }
  .cmdnote { color: var(--muted); font-size: 12px; margin-bottom: 6px; }
  .index-info { font-size: 13px; margin-bottom: 12px; color: var(--muted); }
  .index-info code { background: #f3f4f6; padding: 1px 5px; border-radius: 4px; color: var(--ink); }
  .cmd-box { position: relative; }
  pre { margin: 0; background: #1e1e1e; color: #f1f1f1; border-radius: 8px;
        padding: 14px 46px 14px 14px; overflow-x: auto; font-size: 13px;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; white-space: pre; }
  .copy { position: absolute; top: 8px; right: 8px; border: none; background: #3a3a3a;
          color: #fff; border-radius: 6px; padding: 5px 9px; font-size: 12px; cursor: pointer; }
  .copy:hover { background: var(--orange); }
  .warnmsg { color: var(--bad); font-weight: 600; padding: 6px 0; }
  footer { max-width: 1040px; margin: 0 auto; padding: 8px 20px 40px; color: var(--muted); font-size: 12px; }
  a { color: var(--orange); }
</style>
</head>
<body>
<header>
  <div class="wrap">
    <h1><span>PyTorch</span> install command picker</h1>
    <p>Pick your Python version and GPU, then the torch version and package manager, and get a
       ready-to-paste install command. Generated deterministically from upstream metadata &mdash; __GENERATED__.</p>
  </div>
</header>
<main>
  <div class="card">
    <div class="row">
      <label>Python</label>
      <div class="opts" id="python"></div>
    </div>
    <div class="row">
      <label>GPU vendor</label>
      <div class="opts" id="vendor"></div>
    </div>
    <div class="row">
      <label>Your GPU <span style="font-weight:400;color:var(--muted)">(optional)</span></label>
      <div>
        <div class="combo" id="gpuCombo">
          <input type="text" id="gpuInput" placeholder="Search your GPU… e.g. 4090, A100, H100, 1080 Ti" autocomplete="off" spellcheck="false">
          <button class="clear" id="gpuClear" title="Clear" type="button">&times;</button>
          <div class="combo-list" id="gpuList"></div>
        </div>
        <div class="gpu-info" id="gpuInfo"></div>
        <div class="note-range" id="rangeNote"></div>
      </div>
    </div>
    <div class="row">
      <label>PyTorch version</label>
      <div>
        <select id="version"></select>
        <a id="versionLink" target="_blank" rel="noopener" style="margin-left:10px;font-size:13px;"></a>
        <a id="pypiLink" target="_blank" rel="noopener" style="margin-left:10px;font-size:13px;"></a>
      </div>
    </div>
    <div class="row">
      <label>Operating system</label>
      <div class="opts" id="os"></div>
    </div>
    <div class="row">
      <label>Compute platform</label>
      <div class="opts" id="platform"></div>
    </div>
    <div class="row">
      <label>Package manager</label>
      <div class="opts" id="manager"></div>
    </div>
    <div class="row">
      <label>Packages</label>
      <div class="checks">
        <label><input type="checkbox" id="pkgVision" checked> torchvision</label>
        <label><input type="checkbox" id="pkgAudio" checked> torchaudio</label>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Install command</h2>
    <div class="index-info" id="indexInfo"></div>
    <div id="commands"></div>
  </div>
</main>
<footer>
  NVIDIA matching uses CUDA binary minor-version forward compatibility (a cubin for
  <code>sm_8x</code> runs on a same-major GPU with an equal-or-higher minor) plus PTX JIT forward
  compatibility (<code>+PTX</code> builds run on newer GPUs, with a one-time JIT compile). AMD
  matching uses each card's ROCm support window. When a GPU is selected, only builds it can run are
  shown. torch/CUDA/ROCm data comes from the official PyTorch wheel index and build scripts; GPU
  lists are curated from
  <a href="https://developer.nvidia.com/cuda/gpus">NVIDIA CUDA GPUs</a>
  (<a href="https://developer.nvidia.com/cuda/gpus/legacy">legacy</a>) and the
  <a href="https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html">AMD ROCm support matrix</a>.
</footer>

<script>
const DATA = __DATA__;
const GPUS = __GPUS__;
const OS_LABELS = { linux: "Linux", windows: "Windows", macos: "macOS" };
const MANAGERS = ["pip", "uv", "poetry", "pdm"];
const MANAGER_LABELS = { pip: "pip", uv: "uv", poetry: "Poetry", pdm: "PDM" };
const VENDORS = ["nvidia", "amd"];
const VENDOR_LABELS = { nvidia: "NVIDIA (CUDA)", amd: "AMD (ROCm)" };
const DOCS = {
  nvidia: "https://developer.nvidia.com/cuda/gpus",
  nvidia_legacy: "https://developer.nvidia.com/cuda/gpus/legacy",
  amd_gpus: "https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html",
  amd_gfx: "https://llvm.org/docs/AMDGPUUsage.html#processors",
};
const BASE = "https://download.pytorch.org/whl/";

const state = { python: null, vendor: "nvidia", gpu: "", version: null, os: "linux", platform: null, manager: "pip" };
const $ = (id) => document.getElementById(id);

function ccTuple(s) { const p = s.split("."); return [parseInt(p[0], 10), parseInt(p[1] || "0", 10)]; }
function ccNum(s) { const [a, b] = ccTuple(s); return a * 100 + b; }
function pyKey(p) { const [a, b] = ccTuple(p); return a * 1000 + b; }

// Does a wheel (arch list + PTX flag) run on a GPU of compute capability gpuCc?
function gpuSupport(gpuCc, caps, ptx) {
  if (!caps || caps.length === 0) return { ok: false };
  const [gM, gm] = ccTuple(gpuCc);
  let maxCap = null;
  for (const c of caps) {
    const [aM, am] = ccTuple(c);
    if (aM === gM && am <= gm) return { ok: true, mode: "native" };
    if (maxCap === null || ccNum(c) > ccNum(maxCap)) maxCap = c;
  }
  if (ptx && maxCap && ccNum(gpuCc) >= ccNum(maxCap)) return { ok: true, mode: "ptx" };
  return { ok: false };
}

// ---- selection helpers --------------------------------------------------
function allPythons() {
  const set = new Set();
  DATA.versions.forEach((v) => v.builds.forEach((b) => b.pythons.forEach((p) => set.add(p))));
  return [...set].sort((a, b) => pyKey(a) - pyKey(b));
}
function versionsForPython(py) {
  return DATA.versions.filter((v) => v.builds.some((b) => b.pythons.includes(py)));
}
function getVersion() { return DATA.versions.find((v) => v.version === state.version); }
function osForVersion(vobj, py) {
  const set = new Set();
  vobj.builds.forEach((b) => { if (b.pythons.includes(py)) b.os.forEach((o) => set.add(o)); });
  return [...set];
}
function selectedGpu() { return GPUS.find((g) => g.name === state.gpu) || null; }
function vendorGpus() { return GPUS.filter((g) => g.vendor === state.vendor); }

// Does a PyTorch ROCm wheel of version `ver` support this AMD GPU?
function rocmSupport(gpu, ver) {
  const v = ccNum(ver);
  if (gpu.rocm_min && v < ccNum(gpu.rocm_min)) return false;
  if (gpu.rocm_max && v > ccNum(gpu.rocm_max)) return false;
  return true;
}

// Which backends belong to the selected vendor's compute platforms.
function vendorAllows(backend) {
  if (state.vendor === "nvidia") return backend === "cuda" || backend === "cpu" || backend === "default";
  return backend === "rocm" || backend === "cpu" || backend === "default";
}

function visibleBuilds() {
  const v = getVersion();
  if (!v) return [];
  let builds = v.builds.filter(
    (b) => b.os.includes(state.os) && b.pythons.includes(state.python) && vendorAllows(b.backend)
  );
  const gpu = selectedGpu();
  if (gpu) {
    builds = builds.filter((b) => {
      if (b.backend === "cuda") return gpuSupport(gpu.cc, b.caps, b.ptx).ok;
      if (b.backend === "rocm") return rocmSupport(gpu, b.backend_version);
      return true; // cpu / default
    });
  }
  return builds;
}
function currentBuild() { return visibleBuilds().find((b) => b.key === state.platform) || null; }

function pickDefaultPlatform(builds) {
  const gpu = selectedGpu();
  if (gpu && state.vendor === "nvidia") {
    const cuda = builds.filter((b) => b.backend === "cuda");
    const native = cuda.filter((b) => gpuSupport(gpu.cc, b.caps, b.ptx).mode === "native");
    const pick = native[0] || cuda[0];
    if (pick) return pick.key;
  }
  if (gpu && state.vendor === "amd") {
    const rocm = builds.filter((b) => b.backend === "rocm" && rocmSupport(gpu, b.backend_version));
    if (rocm[0]) return rocm[0].key;
  }
  const dflt = builds.find((b) => b.pypi_default);
  if (dflt) return dflt.key;
  return builds[0] ? builds[0].key : null;
}

// ---- render -------------------------------------------------------------
function optButton(box, label, active, onClick, badgeHtml) {
  const el = document.createElement("div");
  el.className = "opt" + (active ? " active" : "");
  el.innerHTML = label + (badgeHtml || "");
  el.onclick = onClick;
  box.appendChild(el);
}

function renderPython() {
  const box = $("python"); box.innerHTML = "";
  allPythons().forEach((p) => {
    optButton(box, "Python " + p, p === state.python, () => { state.python = p; recompute(); });
  });
}

function renderVersion() {
  const sel = $("version"); sel.innerHTML = "";
  const versions = versionsForPython(state.python);
  versions.forEach((v, i) => {
    const o = document.createElement("option");
    o.value = v.version;
    o.textContent = "torch " + v.version + (i === 0 ? " (latest)" : "") +
      (v.released ? "  ·  " + v.released : "");
    sel.appendChild(o);
  });
  if (state.version) sel.value = state.version;
  const link = $("versionLink");
  const pypi = $("pypiLink");
  if (state.version) {
    link.href = "https://github.com/pytorch/pytorch/releases/tag/v" + state.version;
    link.textContent = "release notes on GitHub ↗";
    link.style.display = "";
    const vobj = getVersion();
    if (vobj && vobj.on_pypi) {
      pypi.href = "https://pypi.org/project/torch/" + state.version + "/";
      pypi.textContent = "PyPI ↗";
      pypi.style.display = "";
    } else {
      pypi.style.display = "none";
    }
  } else {
    link.style.display = "none";
    pypi.style.display = "none";
  }
}

function renderOs() {
  const box = $("os"); box.innerHTML = "";
  const vobj = getVersion(); if (!vobj) return;
  const oses = osForVersion(vobj, state.python);
  ["linux", "windows", "macos"].forEach((o) => {
    if (!oses.includes(o)) return;
    optButton(box, OS_LABELS[o], o === state.os, () => { state.os = o; recompute(); });
  });
}

function renderPlatform() {
  const box = $("platform"); box.innerHTML = "";
  const builds = visibleBuilds();
  if (builds.length === 0) {
    box.innerHTML = '<span class="warnmsg">No matching build &mdash; try a different Python, GPU or OS.</span>';
    return;
  }
  const gpu = selectedGpu();
  builds.forEach((b) => {
    let badge = "";
    if (gpu && b.backend === "cuda") {
      const s = gpuSupport(gpu.cc, b.caps, b.ptx);
      if (s.ok && s.mode !== "native") {
        badge = ' <span class="badge ptx" title="Runs via PTX JIT on first use">PTX</span>';
      }
    }
    optButton(box, b.label, b.key === state.platform, () => { state.platform = b.key; recompute(); }, badge);
  });
}

function renderManager() {
  const box = $("manager"); box.innerHTML = "";
  MANAGERS.forEach((m) => {
    optButton(box, MANAGER_LABELS[m], m === state.manager, () => {
      state.manager = m; renderManager(); renderCommands();
    });
  });
}

function renderVendor() {
  const box = $("vendor"); box.innerHTML = "";
  VENDORS.forEach((v) => {
    optButton(box, VENDOR_LABELS[v], v === state.vendor, () => {
      if (v === state.vendor) return;
      state.vendor = v;
      state.gpu = "";
      $("gpuInput").value = "";
      $("gpuInput").placeholder = v === "amd"
        ? "Search your GPU… e.g. 7900 XTX, MI300X, W7900"
        : "Search your GPU… e.g. 4090, A100, H100, 1080 Ti";
      recompute();
    });
  });
}

// ---- searchable GPU combobox -------------------------------------------
function filteredGpus() {
  const q = $("gpuInput").value.trim().toLowerCase();
  const base = vendorGpus();
  if (!q) return base;
  return base.filter((g) => (g.name + " " + g.cc + " " + g.gfx + " " + g.arch).toLowerCase().includes(q));
}

function renderGpuList() {
  const box = $("gpuList"); box.innerHTML = "";
  const none = document.createElement("div");
  none.className = "combo-item" + (state.gpu === "" ? " sel" : "");
  none.textContent = "— none / pick the platform manually —";
  none.onmousedown = (e) => { e.preventDefault(); selectGpu(""); };
  box.appendChild(none);

  let lastArch = null;
  filteredGpus().forEach((g) => {
    if (g.arch !== lastArch) {
      const hdr = document.createElement("div");
      hdr.className = "combo-group"; hdr.textContent = g.arch;
      box.appendChild(hdr); lastArch = g.arch;
    }
    const meta = g.vendor === "amd" ? g.gfx : "CC " + g.cc;
    const it = document.createElement("div");
    it.className = "combo-item" + (g.name === state.gpu ? " sel" : "");
    it.innerHTML = "<span>" + g.name + '</span><span class="ci-meta">' + meta + "</span>";
    it.onmousedown = (e) => { e.preventDefault(); selectGpu(g.name); };
    box.appendChild(it);
  });
  if (filteredGpus().length === 0) {
    const empty = document.createElement("div");
    empty.className = "combo-item"; empty.style.color = "var(--muted)";
    empty.textContent = "No GPU matches — check the spelling or add it to gpus.py";
    box.appendChild(empty);
  }
}

function showGpuList() { $("gpuList").style.display = "block"; renderGpuList(); }
function hideGpuList() { $("gpuList").style.display = "none"; $("gpuInput").value = state.gpu; }
function selectGpu(name) {
  state.gpu = name;
  if (name) state.platform = null;  // re-pick a GPU-appropriate platform
  $("gpuInput").value = name;
  hideGpuList();
  recompute();
}

function renderGpuInfo() {
  const info = $("gpuInfo"); const gpu = selectedGpu();
  if (!gpu) {
    const kind = state.vendor === "amd" ? "ROCm" : "CUDA";
    info.innerHTML = "Optional &mdash; pick your GPU to filter the compute platforms to the " + kind +
      " builds it can run and hide the rest.";
    return;
  }
  const b = currentBuild();
  if (gpu.vendor === "nvidia") {
    let msg = "<b>" + gpu.name + "</b> — " + gpu.arch + ", compute capability <b>" + gpu.cc + "</b> " +
      '(<a href="' + DOCS.nvidia + '" target="_blank" rel="noopener">NVIDIA docs ↗</a>, ' +
      '<a href="' + DOCS.nvidia_legacy + '" target="_blank" rel="noopener">legacy ↗</a>).';
    const cudaVisible = visibleBuilds().some((x) => x.backend === "cuda");
    if (b && b.backend === "cuda") {
      const s = gpuSupport(gpu.cc, b.caps, b.ptx);
      if (s.ok && s.mode !== "native") {
        msg += ' <span class="pill ptx">PTX JIT</span> No exact SM binary; embedded PTX is JIT-compiled on first run.';
      }
    } else if (!cudaVisible) {
      msg += ' <span class="pill no">no CUDA wheel</span> No CUDA build for torch ' + state.version +
        " supports CC " + gpu.cc + " — pick a different torch version, or use CPU.";
    }
    info.innerHTML = msg;
  } else {
    const supportWindow = "ROCm " + gpu.rocm_min + (gpu.rocm_max ? "–" + gpu.rocm_max : " and later");
    let msg = "<b>" + gpu.name + "</b> — " + gpu.arch + ", LLVM target <b>" + gpu.gfx + "</b> " +
      '(<a href="' + DOCS.amd_gfx + '" target="_blank" rel="noopener">gfx targets ↗</a>). ' +
      "Officially supported on " + supportWindow +
      ' (<a href="' + DOCS.amd_gpus + '" target="_blank" rel="noopener">AMD support matrix ↗</a>).';
    const rocmVisible = visibleBuilds().some((x) => x.backend === "rocm");
    if (!rocmVisible) {
      msg += ' <span class="pill no">no ROCm wheel</span> No ROCm build for torch ' + state.version +
        " matches this GPU — pick a different torch version, or use CPU.";
    }
    info.innerHTML = msg;
  }
}

// --- version "range" note ------------------------------------------------
function ccArchName(cc) { const g = GPUS.find((x) => x.vendor === "nvidia" && x.cc === cc); return g ? g.arch : null; }
function versionCcRange() {
  const v = getVersion(); if (!v) return null;
  const caps = []; let ptx = false;
  v.builds.forEach((b) => {
    if (b.backend === "cuda") { b.caps.forEach((c) => caps.push(c)); if (b.ptx) ptx = true; }
  });
  if (!caps.length) return null;
  caps.sort((a, b) => ccNum(a) - ccNum(b));
  return { min: caps[0], max: caps[caps.length - 1], ptx };
}
function versionRocmRange() {
  const v = getVersion(); if (!v) return null;
  const vers = v.builds.filter((b) => b.backend === "rocm").map((b) => b.backend_version);
  if (!vers.length) return null;
  vers.sort((a, b) => ccNum(a) - ccNum(b));
  return { min: vers[0], max: vers[vers.length - 1] };
}
function renderRangeNote() {
  const el = $("rangeNote");
  if (state.vendor === "nvidia") {
    const r = versionCcRange();
    if (!r) { el.style.display = "none"; return; }
    el.style.display = "block";
    const minA = ccArchName(r.min), maxA = ccArchName(r.max);
    let s = "torch <b>" + state.version + "</b> CUDA wheels are built for NVIDIA compute capabilities <b>" +
      r.min + (minA ? " (" + minA + ")" : "") + "</b> to <b>" + r.max + (maxA ? " (" + maxA + ")" : "") + "</b>. ";
    s += "GPUs <b>older than CC " + r.min + "</b> have no supported wheel for this release; ";
    s += r.ptx
      ? "much newer GPUs still run via PTX JIT. "
      : "GPUs <b>newer than CC " + r.max + "</b> need a later torch release. ";
    s += 'If your card isn\'t shown as supported, that\'s expected — not a bug. Source: ' +
      '<a href="' + DOCS.nvidia + '" target="_blank" rel="noopener">NVIDIA CUDA GPUs ↗</a> ' +
      '(<a href="' + DOCS.nvidia_legacy + '" target="_blank" rel="noopener">legacy ↗</a>).';
    el.innerHTML = s;
  } else {
    const r = versionRocmRange();
    if (!r) {
      el.style.display = "block";
      el.innerHTML = "torch <b>" + state.version + "</b> has no ROCm (AMD) wheels — use a version that ships " +
        "ROCm builds, or select CPU. Source: " +
        '<a href="' + DOCS.amd_gpus + '" target="_blank" rel="noopener">AMD ROCm support matrix ↗</a>.';
      return;
    }
    el.style.display = "block";
    let s = "torch <b>" + state.version + "</b> ships ROCm wheels for <b>ROCm " + r.min +
      (r.min !== r.max ? "–" + r.max : "") + "</b>. ";
    s += "Only AMD GPUs on ROCm's official support list have wheels — older cards dropped by ROCm, " +
      "and cards newer than this ROCm, are unavailable for this release. That's expected, not a bug. Source: " +
      '<a href="' + DOCS.amd_gpus + '" target="_blank" rel="noopener">AMD ROCm support matrix ↗</a>.';
    el.innerHTML = s;
  }
}

// ---- command generation -------------------------------------------------
function pkgList() {
  const list = ["torch==" + state.version];
  if ($("pkgVision").checked) list.push("torchvision");
  if ($("pkgAudio").checked) list.push("torchaudio");
  return list;
}
function pkgNames() { return pkgList().map((p) => p.split("==")[0]); }

function managerBlocks() {
  const b = currentBuild(); if (!b) return [];
  const pkgs = pkgList().join(" ");
  const names = pkgNames();
  const channel = b.channel;           // cu129 / cpu / rocm6.2 / xpu / null
  const url = channel ? BASE + channel : null;
  const isDefaultPypi = b.pypi_default && state.os === "linux";
  // When PyPI serves this exact build (mac default, or the Linux pip-default CUDA
  // build), no custom index / source is needed — the plain command is the command.
  const usePlain = isDefaultPypi || !url;
  const m = state.manager;
  const blocks = [];

  if (m === "pip") {
    blocks.push({ code: usePlain ? "pip install " + pkgs : "pip install " + pkgs + " --index-url " + url });
  } else if (m === "uv") {
    if (usePlain) {
      blocks.push({ label: "uv pip", code: "uv pip install " + pkgs });
      blocks.push({ label: "uv project", code: "uv add " + pkgs });
    } else if (b.backend === "cuda") {
      blocks.push({ label: "Quick (uv pip)", code: "uv pip install " + pkgs + " --torch-backend=" + channel });
      const toml =
        "[[tool.uv.index]]\n" +
        'name = "pytorch"\n' +
        'url = "' + url + '"\n' +
        "explicit = true\n\n" +
        "[tool.uv.sources]\n" +
        names.map((n) => n + ' = { index = "pytorch" }').join("\n");
      blocks.push({ label: "uv project — add to pyproject.toml", code: toml });
      blocks.push({ label: "then", code: "uv add " + pkgs });
    } else if (b.backend === "cpu") {
      blocks.push({ code: "uv pip install " + pkgs + " --torch-backend=cpu" });
    } else {
      blocks.push({ code: "uv pip install " + pkgs + " --index-url " + url });
    }
  } else if (m === "poetry") {
    if (usePlain) {
      blocks.push({ code: "poetry add " + pkgs });
    } else {
      blocks.push({
        code: "poetry source add --priority explicit pytorch " + url + "\n" +
              "poetry add " + pkgs + " --source pytorch",
      });
    }
  } else if (m === "pdm") {
    if (usePlain) {
      blocks.push({ code: "pdm add " + pkgs });
    } else {
      const toml =
        "[[tool.pdm.source]]\n" +
        'name = "pytorch"\n' +
        'url = "' + url + '"\n' +
        'type = "index"\n' +
        "include_packages = [" + names.map((n) => '"' + n + '"').join(", ") + "]";
      blocks.push({ label: "pdm project — add to pyproject.toml", code: toml });
      blocks.push({ label: "then", code: "pdm add " + pkgs });
    }
  }
  return blocks;
}

function renderIndexInfo(b) {
  const el = $("indexInfo");
  if (!b) { el.innerHTML = ""; return; }
  const pypi = '<a href="https://pypi.org/project/torch/' + state.version +
    '/" target="_blank" rel="noopener"><code>PyPI</code> ↗</a>';
  const isDefaultPypi = b.pypi_default && state.os === "linux";
  if (isDefaultPypi) {
    const url = BASE + b.channel;
    el.innerHTML = "This is the default build a plain install gets on Linux — " + pypi +
      ' serves it (same wheels as <a href="' + url + '" target="_blank" rel="noopener"><code>' +
      url + "</code> ↗</a>), so no extra index or source is needed.";
  } else if (b.channel) {
    const url = BASE + b.channel;
    el.innerHTML = 'Wheel index for this build: <a href="' + url + '" target="_blank" rel="noopener"><code>' +
      url + "</code> ↗</a>";
  } else {
    el.innerHTML = "Installs from " + pypi + ".";
  }
}

function renderCommands() {
  const box = $("commands"); box.innerHTML = "";
  const b = currentBuild();
  renderIndexInfo(b);
  if (!b) { box.innerHTML = '<div class="warnmsg">No compatible build for this selection.</div>'; return; }
  managerBlocks().forEach((blk) => {
    const wrap = document.createElement("div"); wrap.className = "cmd";
    if (blk.label) { const h = document.createElement("h3"); h.textContent = blk.label; wrap.appendChild(h); }
    if (blk.note) { const n = document.createElement("div"); n.className = "cmdnote"; n.textContent = blk.note; wrap.appendChild(n); }
    const boxc = document.createElement("div"); boxc.className = "cmd-box";
    const pre = document.createElement("pre"); pre.textContent = blk.code;
    const btn = document.createElement("button"); btn.className = "copy"; btn.textContent = "Copy";
    btn.onclick = () => {
      navigator.clipboard.writeText(blk.code).then(() => {
        btn.textContent = "Copied!"; setTimeout(() => (btn.textContent = "Copy"), 1200);
      });
    };
    boxc.appendChild(pre); boxc.appendChild(btn); wrap.appendChild(boxc); box.appendChild(wrap);
  });
}

// ---- orchestration ------------------------------------------------------
function recompute() {
  const pys = allPythons();
  if (!pys.includes(state.python)) state.python = pys[pys.length - 1];

  const versions = versionsForPython(state.python);
  if (!versions.some((v) => v.version === state.version)) {
    state.version = versions.length ? versions[0].version : null;
  }

  const vobj = getVersion();
  if (vobj) {
    const oses = osForVersion(vobj, state.python);
    if (!oses.includes(state.os)) state.os = oses.includes("linux") ? "linux" : oses[0];
  }

  const vis = visibleBuilds();
  if (!vis.some((b) => b.key === state.platform)) state.platform = pickDefaultPlatform(vis);

  renderPython();
  renderVendor();
  renderVersion();
  renderOs();
  renderPlatform();
  renderManager();
  renderGpuInfo();
  renderRangeNote();
  renderCommands();
}

function init() {
  state.python = allPythons().slice(-1)[0] || null;
  recompute();

  const gi = $("gpuInput");
  gi.onfocus = showGpuList;
  gi.oninput = () => { $("gpuList").style.display = "block"; renderGpuList(); };
  gi.onblur = () => setTimeout(hideGpuList, 150);
  gi.onkeydown = (e) => { if (e.key === "Escape") { gi.blur(); } };
  $("gpuClear").onmousedown = (e) => { e.preventDefault(); selectGpu(""); };

  $("version").onchange = (e) => { state.version = e.target.value; recompute(); };
  ["pkgVision", "pkgAudio"].forEach((id) => { $(id).onchange = renderCommands; });
}

init();
</script>
</body>
</html>
"""
