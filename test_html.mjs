// End-to-end test for the generated interactive picker (docs/index.html).
//
// It loads the self-contained page in jsdom (which actually executes the page's
// JavaScript, so real runtime errors are caught), drives the selectors the way a
// user would, and asserts the resulting install commands and UI state.
//
// Run:  npm install   (once, to get jsdom)
//       npm test      (or: node test_html.mjs)
//
// Regenerate docs/index.html first with `uv run python generate.py` if you have
// changed the generator or the underlying data.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { JSDOM } from "jsdom";

const here = path.dirname(fileURLToPath(import.meta.url));
const html = readFileSync(path.join(here, "docs", "index.html"), "utf-8");

let passed = 0;
const checks = [];
function check(name, fn) { checks.push([name, fn]); }

// Fresh DOM per scenario so tests are independent.
function load() {
  const errors = [];
  const dom = new JSDOM(html, { runScripts: "dangerously" });
  const { window } = dom;
  window.addEventListener("error", (e) => errors.push(String(e.error || e.message)));
  window.navigator.clipboard = { writeText: () => Promise.resolve() };
  return { window, doc: window.document, errors };
}

const norm = (el) => (el ? el.textContent.replace(/\s+/g, " ").trim() : "");
const opts = (doc, sel) => [...doc.querySelectorAll(sel)].map((o) => norm(o));
const active = (doc, sel) => [...doc.querySelectorAll(sel + " .opt.active")].map((o) => norm(o));
const preText = (doc) => [...doc.querySelectorAll("#commands pre")].map((p) => p.textContent).join("\n");
function clickOpt(doc, sel, startsWith) {
  const el = [...doc.querySelectorAll(sel + " .opt")].find((o) => norm(o).startsWith(startsWith));
  assert.ok(el, `option "${startsWith}" not found in ${sel} (have: ${opts(doc, sel + " .opt")})`);
  el.onclick();
}

check("page loads with no JS errors and renders a pip command", () => {
  const { doc, errors } = load();
  assert.deepEqual(errors, [], "unexpected JS errors: " + errors.join("; "));
  assert.equal(active(doc, "#vendor")[0], "NVIDIA (CUDA)");
  assert.ok(doc.getElementById("version").value, "a torch version should be selected");
  assert.ok(active(doc, "#platform").length === 1, "exactly one platform should be active");
  assert.match(preText(doc), /^pip install torch==/m);
});

check("Python is a button group and switching it keeps a valid state", () => {
  const { doc } = load();
  assert.ok(opts(doc, "#python .opt").every((t) => t.startsWith("Python ")));
  const pythons = opts(doc, "#python .opt");
  assert.ok(pythons.length >= 2, "expected several Python buttons");
  clickOpt(doc, "#python", pythons[Math.floor(pythons.length / 2)]);
  assert.equal(active(doc, "#python").length, 1);
});

check("NVIDIA GPU search + selection picks a CUDA build and links to NVIDIA docs + legacy", () => {
  const { window, doc } = load();
  doc.getElementById("gpuInput").value = "4090";
  window.renderGpuList();
  const listed = opts(doc, "#gpuList .combo-item").filter((t) => !t.includes("none"));
  assert.ok(listed.some((t) => t.includes("RTX 4090")), "search '4090' should list the RTX 4090");

  window.selectGpu("GeForce RTX 4090");
  assert.ok(active(doc, "#platform")[0].startsWith("CUDA"), "an NVIDIA GPU should select a CUDA build");
  const info = doc.getElementById("gpuInfo").innerHTML;
  assert.match(info, /compute capability <b>8\.9<\/b>/);
  assert.match(info, /developer\.nvidia\.com\/cuda\/gpus"/, "must link the current NVIDIA docs URL");
  assert.match(info, /developer\.nvidia\.com\/cuda\/gpus\/legacy/, "must link the legacy NVIDIA docs URL");
  assert.doesNotMatch(info, /cuda-gpus/, "old cuda-gpus URL must be gone");
});

check("only CUDA/CPU shown for NVIDIA; unsupported CUDA builds are hidden (not crossed out)", () => {
  const { window, doc } = load();
  window.selectGpu("GeForce GTX 1080 Ti"); // CC 6.1
  const platforms = opts(doc, "#platform .opt");
  assert.ok(!platforms.some((t) => t.startsWith("ROCm")), "NVIDIA vendor must not show ROCm");
  assert.ok(platforms.includes("CPU"), "CPU must always be available");
  assert.ok(!/✗|✕|✘/.test(preText(doc) + platforms.join()), "no cross-out markers");
});

check("switching to AMD shows ROCm platforms and a rocm install command", () => {
  const { doc } = load();
  clickOpt(doc, "#vendor", "AMD");
  assert.equal(active(doc, "#vendor")[0], "AMD (ROCm)");
  const platforms = opts(doc, "#platform .opt");
  assert.ok(platforms.some((t) => t.startsWith("ROCm")), "AMD vendor should offer ROCm");
  assert.ok(!platforms.some((t) => t.startsWith("CUDA")), "AMD vendor must not show CUDA");
  assert.match(preText(doc), /--index-url https:\/\/download\.pytorch\.org\/whl\/rocm/);
});

check("AMD GPU search + selection resolves a compatible ROCm build with source links", () => {
  const { window, doc } = load();
  clickOpt(doc, "#vendor", "AMD");
  doc.getElementById("gpuInput").value = "7900";
  window.renderGpuList();
  const listed = opts(doc, "#gpuList .combo-item").filter((t) => !t.includes("none"));
  assert.ok(listed.some((t) => t.includes("7900 XTX")), "AMD search '7900' should list RX 7900 XTX");

  window.selectGpu("Radeon RX 7900 XTX");
  const info = doc.getElementById("gpuInfo").innerHTML;
  assert.match(info, /gfx1100/, "should show the AMD gfx target");
  assert.match(info, /rocm\.docs\.amd\.com/, "should link the AMD ROCm support matrix");
  assert.match(info, /llvm\.org\/docs\/AMDGPUUsage/, "should link the LLVM gfx targets doc");
  assert.ok(active(doc, "#platform")[0].startsWith("ROCm"), "AMD GPU should select a ROCm build");
});

check("AMD card outside the ROCm window falls back to CPU with a clear message", () => {
  const { window, doc } = load();
  clickOpt(doc, "#vendor", "AMD");
  window.selectGpu("Radeon RX Vega 64"); // dropped after ROCm 5.4; latest torch ships rocm 7.x
  const platforms = opts(doc, "#platform .opt");
  assert.deepEqual(platforms, ["CPU"], "no ROCm build should match; only CPU remains");
  assert.match(doc.getElementById("gpuInfo").innerHTML, /no ROCm wheel/);
});

check("package-manager selection switches the shown command; uv emits an index config for non-default CUDA", () => {
  const { window, doc } = load();
  window.selectGpu("GeForce RTX 4090");
  clickOpt(doc, "#manager", "uv");
  assert.equal(active(doc, "#manager")[0], "uv");
  const text = preText(doc);
  assert.match(text, /uv pip install .*--torch-backend=cu/);
  assert.match(text, /\[\[tool\.uv\.index\]\]/, "non-default CUDA should include a uv index config");
  assert.match(text, /download\.pytorch\.org\/whl\/cu/, "config should point at the pytorch index");
});

check("version links to its PyTorch GitHub release and updates the link", () => {
  const { doc } = load();
  const link = doc.getElementById("versionLink");
  const v = doc.getElementById("version").value;
  assert.equal(link.getAttribute("href"), "https://github.com/pytorch/pytorch/releases/tag/v" + v);
  assert.match(norm(link), /release notes/i);
});

check("the Linux pip-default build uses plain commands for every manager (no index/source)", () => {
  const { doc } = load(); // initial selection is the PyPI default CUDA build on Linux
  assert.match(doc.getElementById("indexInfo").innerHTML, /default build/i);
  const managers = [["pip", "pip"], ["uv", "uv"], ["poetry", "Poetry"], ["pdm", "PDM"]];
  for (const [, label] of managers) {
    clickOpt(doc, "#manager", label);
    const t = preText(doc);
    assert.doesNotMatch(t, /--index-url|--torch-backend|source add|\[\[tool\.(uv|pdm)/,
      label + " should use a plain command for the default build (got: " + t + ")");
    assert.match(t, /(pip install|uv (pip install|add)|poetry add|pdm add) torch==/,
      label + " should still install torch");
  }
});

check("version links to PyPI, and the wheel-index link reflects the selected platform", () => {
  const { window, doc } = load();
  const v = doc.getElementById("version").value;
  const pypi = doc.getElementById("pypiLink");
  assert.equal(pypi.getAttribute("href"), "https://pypi.org/project/torch/" + v + "/");
  assert.match(norm(pypi), /PyPI/);

  window.selectGpu("GeForce RTX 4090"); // CUDA build -> download.pytorch.org index
  const info = doc.getElementById("indexInfo").innerHTML;
  assert.match(info, /download\.pytorch\.org\/whl\/cu\d+/, "index link should point at the cuXX wheel index");
});

check("no user-visible \"native\" label remains in the rendered command panel", () => {
  const { window, doc } = load();
  window.selectGpu("A100");
  // The word may appear in the <script> source, but not in rendered command output.
  assert.doesNotMatch(preText(doc), /native/i);
});

check("header links to the GitHub repo and the Markdown compatibility table", () => {
  const { doc } = load();
  const links = [...doc.querySelectorAll("header .toolbar a.btn")];
  assert.ok(links.length >= 2, "header should have repo + table buttons");
  const repo = links.find((a) => /github repo/i.test(norm(a)));
  const table = links.find((a) => /compatibility table/i.test(norm(a)));
  assert.ok(repo && /github\.com\//.test(repo.getAttribute("href")), "repo button must link to GitHub");
  assert.ok(table && /COMPATIBILITY\.md/.test(table.getAttribute("href")), "table button must link to the .md");
});

check("two tabs exist and switching to 'Search by PyTorch' reveals the torch panel", () => {
  const { doc } = load();
  assert.ok(doc.getElementById("tab-python"), "python tab panel exists");
  assert.ok(doc.getElementById("tab-torch"), "torch tab panel exists");
  assert.equal(doc.getElementById("tab-torch").hidden, true, "torch panel hidden initially");

  doc.getElementById("tabBtnTorch").onclick();
  assert.equal(doc.getElementById("tab-python").hidden, true);
  assert.equal(doc.getElementById("tab-torch").hidden, false);
  assert.ok(doc.getElementById("tabBtnTorch").classList.contains("active"));
  assert.ok(doc.getElementById("tvVersion").value, "a torch version should be selected in tab 2");
});

check("'Search by PyTorch' shows Python, CUDA (caps + GPUs) and AMD/ROCm (gfx + GPUs)", () => {
  const { doc } = load();
  doc.getElementById("tabBtnTorch").onclick();
  const out = doc.getElementById("tvOut");
  const headings = [...out.querySelectorAll(".tv-section h3")].map((h) => norm(h));
  assert.ok(headings.some((t) => /Python versions/i.test(t)), "should list Python versions");
  assert.ok(headings.some((t) => /NVIDIA/i.test(t)), "should have an NVIDIA/CUDA section");
  assert.ok(headings.some((t) => /AMD/i.test(t)), "should have an AMD/ROCm section");

  // CUDA table: a compute-capability cell and a collapsible GPU list.
  const cudaTable = [...out.querySelectorAll(".tv-table")][0];
  assert.ok(cudaTable, "a CUDA table should render");
  assert.match(cudaTable.innerHTML, /\d\.\d/, "CUDA table should show compute capabilities");
  assert.ok(cudaTable.querySelector("details.gpus"), "CUDA rows should list supported GPUs");
  assert.match(cudaTable.textContent, /GeForce RTX 4090|A100|H100/, "CUDA GPUs should be enumerated");

  // The wheel-index link in the CUDA cell points at download.pytorch.org.
  assert.match(cudaTable.innerHTML, /download\.pytorch\.org\/whl\/cu\d+/);
});

check("'Search by PyTorch' version switch re-renders and links to GitHub + PyPI", () => {
  const { doc } = load();
  doc.getElementById("tabBtnTorch").onclick();
  const sel = doc.getElementById("tvVersion");
  const options = [...sel.options].map((o) => o.value);
  assert.ok(options.length >= 2, "several torch versions selectable");
  sel.value = options[1];
  sel.onchange({ target: sel });

  const vlink = doc.getElementById("tvVersionLink");
  assert.equal(vlink.getAttribute("href"), "https://github.com/pytorch/pytorch/releases/tag/v" + options[1]);
  assert.match(doc.getElementById("tvSummary").innerHTML, /Released/);
});

// --- run -----------------------------------------------------------------
let failed = 0;
for (const [name, fn] of checks) {
  try {
    fn();
    passed++;
    console.log("ok   " + name);
  } catch (err) {
    failed++;
    console.error("FAIL " + name + "\n     " + (err && err.message ? err.message : err));
  }
}
console.log(`\n${passed}/${checks.length} checks passed`);
process.exit(failed ? 1 : 0);
