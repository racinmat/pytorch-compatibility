"""Extract supported CUDA compute capabilities from PyTorch build sources.

The compute capabilities (SM architectures) baked into each official CUDA wheel are
defined by ``TORCH_CUDA_ARCH_LIST`` in the manylinux build script, keyed by CUDA version:

    * torch >= 2.6 : pytorch/pytorch  @ v<ver>            :: .ci/manywheel/build_cuda.sh
    * torch <= 2.5 : pytorch/builder  @ release/<maj.min> :: manywheel/build_cuda.sh

This module locates the right script for a torch version and *evaluates* the relevant
``case ${CUDA_VERSION} in ... esac`` block for a concrete CUDA version, faithfully
reproducing the shell semantics we rely on (literal lists, ``${TORCH_CUDA_ARCH_LIST}``
appends, glob labels like ``11.[67]`` / ``10.*``, and the aarch64 / libtorch ``if`` guards
which are false for the standard x86_64 python wheel).
"""

from __future__ import annotations

import ast
import fnmatch
import re
from dataclasses import dataclass

from . import sources

_PY_TABLE_MARK = "TORCH_CUDA_ARCH_LIST_TABLE"

_RAW = "https://raw.githubusercontent.com"


@dataclass(frozen=True)
class ArchResult:
    torch_minor: str        # "2.4"
    cuda_version: str       # "12.1"
    arch_list_raw: str      # "5.0;6.0;7.0;7.5;8.0;8.6;9.0"
    capabilities: tuple[str, ...]  # ("5.0", "6.0", ...) with +PTX stripped
    has_ptx: bool
    source_url: str


def _parse_minor(torch_version: str) -> tuple[int, int]:
    parts = re.split(r"[.\-+]", torch_version)
    return int(parts[0]), int(parts[1])


def candidate_urls(torch_version: str) -> list[str]:
    """Ordered list of build-source URLs to try for a torch version.

    Covers every known upstream location of the CUDA arch-list definition:
      * pytorch/pytorch  @ v<ver>  .ci/manywheel/build_cuda.sh        (2.6 .. 2.12, shell)
      * pytorch/pytorch  @ v<ver>  .ci/manywheel/build_env_setup.py   (new Python-dict form)
      * pytorch/pytorch  @ main    .ci/manywheel/build_env_setup.py   (nightlies / next release)
      * pytorch/builder  @ release/<maj.min> manywheel/build_cuda.sh  (<= 2.5, shell)
    """
    major, minor = _parse_minor(torch_version)
    mm = f"{major}.{minor}"
    pt = f"{_RAW}/pytorch/pytorch"
    # build_cuda.sh in newer builder branches; older branches (<=1.11) keep the arch
    # `case` inside build.sh. build.sh is only accepted if it actually contains a case.
    bld = f"{_RAW}/pytorch/builder/release/{mm}/manywheel/build_cuda.sh"
    bld_alt = f"{_RAW}/pytorch/builder/release/{mm}/manywheel/build.sh"
    tagged = [
        f"{pt}/v{torch_version}/.ci/manywheel/build_cuda.sh",
        f"{pt}/v{mm}.0/.ci/manywheel/build_cuda.sh",
        f"{pt}/v{torch_version}/.ci/manywheel/build_env_setup.py",
        f"{pt}/v{mm}.0/.ci/manywheel/build_env_setup.py",
    ]
    nightly = [
        f"{pt}/main/.ci/manywheel/build_cuda.sh",
        f"{pt}/main/.ci/manywheel/build_env_setup.py",
    ]
    urls = (
        (tagged + [bld, bld_alt] + nightly)
        if (major, minor) >= (2, 6)
        else ([bld, bld_alt] + tagged + nightly)
    )
    seen: set[str] = set()
    return [u for u in urls if not (u in seen or seen.add(u))]


def resolve_script(torch_version: str, *, force: bool = False) -> tuple[str, str] | None:
    """Return (script_text, source_url) for the first candidate that yields an arch list."""
    for url in candidate_urls(torch_version):
        text = sources.try_fetch(url, force=force)
        if text is not None and (_CASE_START.search(text) or _PY_TABLE_MARK in text):
            return text, url
    return None


# --- shell `case` evaluation ------------------------------------------------

_CASE_START = re.compile(r"case\s+\$\{?CUDA_VERSION\}?\s+in\b")
_ASSIGN_MARK = "TORCH_CUDA_ARCH_LIST="


def _extract_rhs(line: str) -> str | None:
    """Return the RHS of a ``TORCH_CUDA_ARCH_LIST=`` assignment on a line.

    Quote-aware: the RHS may be a double-quoted string that itself contains inner
    double quotes inside a ``$(...)`` command substitution, so we take the text between
    the first and last quote rather than a naive ``"[^"]*"`` match.
    """
    i = line.find(_ASSIGN_MARK)
    if i == -1:
        return None
    rest = line[i + len(_ASSIGN_MARK):].strip()
    rest = re.sub(r";;\s*$", "", rest).strip()
    if not rest:
        return None
    if rest.startswith('"'):
        last = rest.rfind('"')
        if last <= 0:
            return None
        return rest[1:last]
    return rest.split("#")[0].split()[0]


def _base_arch_list(lines: list[str], case_idx: int) -> str:
    """Nearest literal (no variable reference) assignment above the case block."""
    for i in range(case_idx - 1, -1, -1):
        val = _extract_rhs(lines[i])
        if val is not None and "$" not in val:
            return val
    return ""


def _split_cases(lines: list[str], case_idx: int) -> list[tuple[list[str], list[str]]]:
    """Return [(labels, body_lines), ...] for the case block starting at case_idx.

    Handles both the multi-line form (``label)`` / body / ``;;`` on separate lines) and
    the single-line form (``label) body ;;``) by treating ``;;`` as the clause delimiter
    and the first ``)`` in a clause as the label/body separator. Bash arch lists use a
    single ``;`` internally, so ``;;`` is an unambiguous delimiter here.
    """
    # Collect the raw text of the case block (between `... in` and `esac`).
    end = next(
        (j for j in range(case_idx, len(lines)) if lines[j].strip() == "esac"),
        len(lines),
    )
    block = "\n".join(lines[case_idx + 1 : end])

    cases: list[tuple[list[str], list[str]]] = []
    for clause in block.split(";;"):
        # Drop whole-line comments so the label is easy to locate.
        cleaned = "\n".join(
            ln for ln in clause.splitlines() if not ln.strip().startswith("#")
        )
        paren = cleaned.find(")")
        if paren == -1:
            continue
        label_part = cleaned[:paren].strip()
        if not label_part:
            continue
        labels = [x.strip() for x in label_part.split("|") if x.strip()]
        body = cleaned[paren + 1 :].splitlines()
        cases.append((labels, body))
    return cases


_CMD_SUBST = re.compile(r"\$\((?P<inner>[^()]*)\)")


def _resolve_value(val: str, cur: str) -> str:
    """Resolve a raw assignment RHS to its value for the default x86_64 python wheel."""
    val = val.replace("${TORCH_CUDA_ARCH_LIST}", cur).replace("$TORCH_CUDA_ARCH_LIST", cur)

    def _sub(match: re.Match) -> str:
        inner = match.group("inner")
        # `[[ ... ]] && echo "A" || echo "B"` -> default variant takes the `||` branch.
        alt = re.search(r"\|\|\s*echo\s+\"?([^\"]*)\"?", inner)
        return alt.group(1).strip() if alt else ""

    val = _CMD_SUBST.sub(_sub, val)
    return val


def _apply_body(body: list[str], base: str) -> str:
    """Evaluate arch-list assignments in a case body for the *default* variant.

    The aarch64 / libtorch ``if`` guards are all false for the standard x86_64 python
    wheel, so every ``then`` branch is skipped and any ``else`` branch is taken.
    """
    cur = base
    active = True
    stack: list[bool] = []
    for line in body:
        s = line.strip()
        if s.startswith("if ") or s.startswith("if["):
            stack.append(active)
            active = False  # guard condition is false for the default wheel
            continue
        if s == "else":
            active = stack[-1] if stack else True
            continue
        if s in ("fi", "fi;"):
            active = stack.pop() if stack else True
            continue
        if not active:
            continue
        val = _extract_rhs(s)
        if val is not None:
            cur = _resolve_value(val, cur)
    return cur


def _evaluate_case(script_text: str, cuda_version: str) -> str | None:
    lines = script_text.splitlines()
    case_idx = next((i for i, ln in enumerate(lines) if _CASE_START.search(ln)), None)
    if case_idx is None:
        return None
    base = _base_arch_list(lines, case_idx)
    for labels, body in _split_cases(lines, case_idx):
        if "*" in labels:  # the "unknown cuda version -> exit 1" default branch
            continue
        if any(fnmatch.fnmatchcase(cuda_version, lbl) for lbl in labels):
            return _apply_body(body, base)
    return None


def _extract_dict_literal(text: str, name: str) -> dict | None:
    """Parse the ``name = { ... }`` dict literal from ``text`` via a brace scan + ast."""
    m = re.search(rf"{re.escape(name)}\s*(?::[^=]+)?=\s*", text)
    if not m:
        return None
    start = text.find("{", m.end())
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return ast.literal_eval(text[start : i + 1])
                except (ValueError, SyntaxError):
                    return None
    return None


def _evaluate_python_table(
    script_text: str, cuda_version: str, arch: str = "x86_64"
) -> str | None:
    """Evaluate the new ``TORCH_CUDA_ARCH_LIST_TABLE`` dict form (release wheel = no +PTX)."""
    table = _extract_dict_literal(script_text, _PY_TABLE_MARK)
    if not table or cuda_version not in table:
        return None
    archs = table[cuda_version].get(arch)
    if not archs:
        return None
    return ";".join(f"{cc // 10}.{cc % 10}" for cc in sorted(archs))


def evaluate(script_text: str, cuda_version: str) -> str | None:
    """Return the raw ``TORCH_CUDA_ARCH_LIST`` for ``cuda_version`` (e.g. "12.1"), or None.

    Handles both upstream formats: the shell ``case ${CUDA_VERSION}`` block and the newer
    ``TORCH_CUDA_ARCH_LIST_TABLE`` Python dict in build_env_setup.py.
    """
    return _evaluate_case(script_text, cuda_version) or _evaluate_python_table(
        script_text, cuda_version
    )


def _to_capabilities(raw: str) -> tuple[tuple[str, ...], bool]:
    caps: list[str] = []
    has_ptx = False
    for token in raw.split(";"):
        token = token.strip()
        if not token:
            continue
        if token.endswith("+PTX"):
            has_ptx = True
            token = token[: -len("+PTX")]
        token = token.strip()
        if re.fullmatch(r"\d+\.\d+[a-z]?", token):  # e.g. 9.0, 10.0, 12.0, 9.0a
            caps.append(token)
    # de-dupe, keep numeric order
    uniq = sorted(set(caps), key=lambda s: [int(re.sub(r"\D", "", x) or 0) for x in s.split(".")])
    return tuple(uniq), has_ptx


def arch_result(
    torch_version: str, cuda_version: str, *, force: bool = False
) -> ArchResult | None:
    resolved = resolve_script(torch_version, force=force)
    if resolved is None:
        return None
    script_text, url = resolved
    raw = evaluate(script_text, cuda_version)
    if raw is None:
        return None
    caps, has_ptx = _to_capabilities(raw)
    major, minor = _parse_minor(torch_version)
    return ArchResult(
        torch_minor=f"{major}.{minor}",
        cuda_version=cuda_version,
        arch_list_raw=raw,
        capabilities=caps,
        has_ptx=has_ptx,
        source_url=url,
    )
