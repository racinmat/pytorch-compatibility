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

// Fresh DOM per scenario so tests are independent. Pass a config object to
// simulate a different deployment (e.g. the internal Artifactory mirror) by
// rewriting the embedded CFG — this is exactly what generate.py --config does.
function load(cfg) {
  const errors = [];
  let src = html;
  if (cfg) {
    const replaced = src.replace(/const CFG = \{[^\n]*?\};/, "const CFG = " + JSON.stringify(cfg) + ";");
    assert.notEqual(replaced, src, "failed to substitute CFG in the page");
    src = replaced;
  }
  const dom = new JSDOM(src, { runScripts: "dangerously" });
  const { window } = dom;
  window.addEventListener("error", (e) => errors.push(String(e.error || e.message)));
  window.navigator.clipboard = { writeText: () => Promise.resolve() };
  return { window, doc: window.document, errors };
}

const ARTIFACTORY = "https://artifacts.int.example.com/artifactory/api/pypi/pytorch-whl-";
const INTERNAL_CFG = {
  mode: "internal",
  default_index_base: "https://download.pytorch.org/whl/",
  index_overrides: {
    cu126: ARTIFACTORY + "cu126-remote/simple/",
    cpu: ARTIFACTORY + "cpu-remote/simple/",
    "rocm7.2": ARTIFACTORY + "rocm7.2-remote/simple/",
  },
  missing_index_message:
    'The <code>pytorch-whl-{channel}-remote</code> index is not mirrored yet — ' +
    '<a href="https://jira.example/secure/CreateIssueDetails!init.jspa?pid=1&amp;summary=Mirror%20pytorch-whl-{channel}-remote">open a request</a> ' +
    'or clone <a href="https://jira.example/browse/BLD-11022">BLD-11022</a>.',
};
const platBtn = (doc, startsWith) =>
  [...doc.querySelectorAll("#platform .opt")].find((o) => norm(o).startsWith(startsWith));

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

check("uv shows the pyproject.toml index snippet for every non-default backend (e.g. CPU), not just CUDA", () => {
  const { doc } = load();
  platBtn(doc, "CPU").onclick(); // CPU is a non-default index on Linux
  clickOpt(doc, "#manager", "uv");
  const t = preText(doc);
  assert.match(t, /uv pip install .*--torch-backend=cpu/, "quick command still uses --torch-backend=cpu");
  assert.match(t, /\[\[tool\.uv\.index\]\]/, "CPU (non-default) must also show the uv pyproject snippet");
  assert.match(t, /download\.pytorch\.org\/whl\/cpu/, "snippet url points at the cpu index");
  assert.match(t, /\[tool\.uv\.sources\][\s\S]*torch = \{ index = "pytorch-cpu" \}/,
    "index name should reflect the channel (pytorch-cpu)");
  assert.match(t, /uv add torch==/, "should include the 'then: uv add' step");
});

check("Poetry emits a pyproject.toml source section (like uv) for non-default CUDA", () => {
  const { window, doc } = load();
  window.selectGpu("GeForce RTX 4090"); // non-default CUDA build
  clickOpt(doc, "#manager", "Poetry");
  const t = preText(doc);
  assert.match(t, /\[\[tool\.poetry\.source\]\]/, "should include a poetry source section");
  assert.match(t, /priority = "explicit"/);
  assert.match(t, /\[tool\.poetry\.dependencies\]/);
  assert.match(t, /torch = \{ version = "[^"]+", source = "pytorch-cu\d+" \}/, "torch dep must be pinned to the channel-named source");
  assert.match(t, /download\.pytorch\.org\/whl\/cu\d+/, "source url must be the pytorch index");
  // The CLI form is still offered too, with the channel-reflecting source name.
  assert.match(t, /poetry source add --priority explicit pytorch-cu\d+ /);
  assert.match(t, /poetry add .*--source pytorch-cu\d+/);
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

check("range note is per-build accurate: default wheel floor + binary forward-compat, no false ceiling", () => {
  const { doc } = load(); // latest torch version selected by default
  const note = norm(doc.getElementById("rangeNote"));
  // Must not repeat the old bug: claiming same-major newer GPUs need a later release.
  assert.doesNotMatch(note, /newer than CC[\s\S]*need a later torch release/i,
    "must not claim same-major newer cards need a later release (got: " + note + ")");
  // Must explain binary forward-compatibility (so e.g. sm_121 on the sm_120 build).
  assert.match(note, /binary forward-compatibility/i);
  // Only a newer architecture *major* should require a later release.
  assert.match(note, /architecture major newer than/i);
  // Multi-build versions should surface the default pip wheel and its floor.
  if (/differ by CUDA version/i.test(note)) {
    assert.match(note, /plain .*pip install.* pulls the CUDA/i, "should name the default pip wheel");
    assert.match(note, /install it explicitly/i, "should tell users to install an older CUDA build for old cards");
  }
  assert.match(note, /CUDA binary compatibility/i, "should link the CUDA binary-compatibility docs");
});

check("DGX Spark (CC 12.1) runs on a CC 12.0 CUDA build via minor-version forward-compat", () => {
  const { window, doc } = load();
  doc.getElementById("gpuInput").value = "DGX";
  window.renderGpuList();
  const listed = opts(doc, "#gpuList .combo-item").filter((t) => !t.includes("none"));
  assert.ok(listed.some((t) => /DGX Spark/.test(t)), "search 'DGX' should list the GB10 / DGX Spark");

  window.selectGpu("GB10 (Grace Blackwell, DGX Spark)");
  assert.ok(active(doc, "#platform")[0].startsWith("CUDA"),
    "a CC 12.1 card must resolve to a CUDA build (not CPU) via forward-compat");
  assert.match(doc.getElementById("gpuInfo").innerHTML, /compute capability <b>12\.1<\/b>/);
  // It is a native (binary) match on the 12.0 cubin, so no PTX-JIT warning should show.
  assert.doesNotMatch(doc.getElementById("gpuInfo").innerHTML, /PTX JIT/);
});

check("internal config: a mirrored channel (cu126) emits the Artifactory index for pip and uv", () => {
  const { window, doc } = load(INTERNAL_CFG);
  window.selectGpu("GeForce RTX 4090");
  platBtn(doc, "CUDA 12.6").onclick();
  assert.match(preText(doc), new RegExp("pip install .*--index-url " + ARTIFACTORY.replace(/[.]/g, "\\.") + "cu126-remote/simple/"));

  clickOpt(doc, "#manager", "uv");
  const t = preText(doc);
  // uv must NOT use --torch-backend for a mirrored index; it must use the explicit URL + config.
  assert.doesNotMatch(t, /--torch-backend/);
  assert.match(t, /--index-url .*cu126-remote/);
  assert.match(t, /\[\[tool\.uv\.index\]\][\s\S]*cu126-remote/);
});

check("internal config: a non-mirrored channel shows the placeholder message (HTML links, {channel} filled), no command", () => {
  const { window, doc } = load(INTERNAL_CFG);
  window.selectGpu("GeForce RTX 4090");
  platBtn(doc, "CUDA 12.9").onclick(); // cu129 is not in index_overrides
  assert.equal(preText(doc), "", "no install command should render for an unconfigured index");

  const note = doc.querySelector("#commands .missing-note");
  assert.ok(note, "a missing-index note should render");
  // The message is rendered as HTML: links become real anchors, not plaintext.
  const links = [...note.querySelectorAll("a")];
  assert.ok(links.length >= 2, "message links must render as <a> elements");
  assert.ok(links.some((a) => /BLD-11022/.test(a.getAttribute("href"))), "clone-ticket link present");
  // {channel} is substituted with the selected channel everywhere (text + URLs).
  assert.match(note.innerHTML, /pytorch-whl-cu129-remote/);
  assert.doesNotMatch(note.innerHTML, /\{channel\}/, "no unsubstituted {channel} token left");
  assert.ok(links.some((a) => /summary=Mirror%20pytorch-whl-cu129-remote/.test(a.getAttribute("href"))),
    "create-ticket URL should carry the channel-filled summary");
  assert.match(doc.getElementById("indexInfo").innerHTML, /not configured/i);
});

check("internal config: the Linux pip-default build still works via a plain command", () => {
  const { doc } = load(INTERNAL_CFG); // default 2.13 build is cu130 (not mirrored) but PyPI-served
  const t = preText(doc);
  assert.match(t, /^pip install torch==/m);
  assert.doesNotMatch(t, /--index-url|artifactory/);
});

check("internal config: CPU and ROCm mirrored channels resolve to Artifactory", () => {
  const { window, doc } = load(INTERNAL_CFG);
  window.selectGpu("GeForce RTX 4090");
  platBtn(doc, "CPU").onclick();
  assert.match(preText(doc), /--index-url .*cpu-remote\/simple\//);

  clickOpt(doc, "#vendor", "AMD");
  window.selectGpu("Radeon RX 7900 XTX");
  assert.ok(active(doc, "#platform")[0].startsWith("ROCm 7.2"), "AMD card should pick the mirrored ROCm 7.2");
  assert.match(preText(doc), /--index-url .*rocm7\.2-remote\/simple\//);
});

check("public config (default): commands use download.pytorch.org, never Artifactory", () => {
  const { window, doc } = load();
  window.selectGpu("GeForce RTX 4090");
  clickOpt(doc, "#manager", "uv");
  const t = preText(doc);
  assert.doesNotMatch(t, /artifactory/i);
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
