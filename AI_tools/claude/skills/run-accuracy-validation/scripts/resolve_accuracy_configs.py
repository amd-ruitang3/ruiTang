#!/usr/bin/env python3
"""Resolve the CI accuracy-validation run plan for a given model.

`.github/benchmark/models_accuracy.json` is the source of truth for *which*
accuracy permutations CI runs. Each entry is one fully-specified run:
(model_path, extraArgs, env_vars, test_level, accuracy_threshold/baseline,
optional client_command). "Run all accuracy tests for DeepSeek-V4" means
"run every catalog entry whose model matches, at every test_level".

This script does the one piece of glue that is easy to get wrong by hand:
translate a catalog entry into the exact arguments the run-atom-workload
scripts expect —

  - start_atom_server.sh wants TP as a *positional* arg, so `-tp N` must be
    lifted out of extraArgs and the remainder passed as EXTRA_ARGS.
  - env_vars is newline-separated in the JSON; the shell wants a
    `VAR=val VAR=val` prefix.
  - a subset of entries carry a custom client_command (chat-completions
    lm_eval) that overrides the default local-completions gsm8k client.
  - simple_inference is an *offline* smoke test that reuses the same env +
    a subset of the server args (serving-throughput flags dropped).

By default the *locally checked-out* catalog is read, so freshness follows your
git state. Pass --from-remote to read the catalog from a git ref instead (e.g.
origin/main) — the script fetches that ref first, so you get the latest CI matrix
without having to `git pull` your working branch.

Usage:
  resolve_accuracy_configs.py MODEL_FILTER [--test-level pr,nightly,main]
                              [--format table|json]
                              [--catalog PATH | --from-remote [REF]]

MODEL_FILTER is a case-insensitive substring matched against both model_name
and model_path (e.g. "V4", "deepseek-v4", "R1-0528"). Omit --test-level to
include every level (the default — "run all related").
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

# Repo-relative location of the catalog — used for both the local default and the
# `git show <ref>:<path>` remote read so the two never drift apart.
CATALOG_REL_PATH = ".github/benchmark/models_accuracy.json"
DEFAULT_REMOTE_REF = "origin/main"

# Catalog fields have drifted across two spellings; tolerate both.
_EXTRA_ARGS_KEYS = ("extraArgs", "extra_args")
_THRESHOLD_KEYS = ("accuracy_threshold", "accuracy_test_threshold")

# gsm8k client defaults — CI uses 3-shot flexible-extract; concurrency 65 is the
# value baked into run_gsm8k_eval.sh. Kept here so the plan is self-describing.
DEFAULT_FEWSHOT = 3
DEFAULT_NUM_CONCURRENT = 65

# TP flag spellings seen in extraArgs.
_TP_FLAGS = ("-tp", "--tp", "--tensor-parallel-size", "-tensor-parallel-size")

# Serving-throughput / scheduling flags that only make sense for the online
# server. simple_inference is a load-and-generate smoke test, so these are
# dropped from the offline arg list. Flags that take a value are removed
# together with their value. Everything NOT listed here is kept verbatim, so
# quant/spec/model-shape flags (--kv_cache_dtype, --method, --hf-overrides,
# --online_quant_config, --max-model-len, …) survive into the offline run.
_OFFLINE_DROP_FLAGS_WITH_VALUE = {
    "--gpu-memory-utilization",
    "--max-num-batched-tokens",
}
_OFFLINE_DROP_BOOL_FLAGS = {
    "--enable-dp-attention",
    "--enable-tbo",
    "--no-enable_prefix_caching",
}


def _get(entry: dict, keys: tuple[str, ...], default: str = "") -> str:
    for k in keys:
        if k in entry and entry[k] not in (None, ""):
            return entry[k]
    return default


def _split_tp(extra_args: str) -> tuple[int | None, str]:
    """Lift the TP value out of extraArgs. Returns (tp_or_None, remaining_args).

    Handles `-tp 8`, `--tp=8`, `--tensor-parallel-size 8`. Quoting (e.g. inside
    --hf-overrides '{...}') is respected via shlex so we never split a JSON blob.
    """
    tokens = shlex.split(extra_args)
    tp: int | None = None
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        matched = False
        for flag in _TP_FLAGS:
            if tok == flag:  # "-tp", "8"
                if i + 1 < len(tokens):
                    tp = int(tokens[i + 1])
                    i += 2
                    matched = True
                break
            if tok.startswith(flag + "="):  # "--tp=8"
                tp = int(tok.split("=", 1)[1])
                i += 1
                matched = True
                break
        if matched:
            continue
        out.append(tok)
        i += 1
    return tp, shlex.join(out)


def _to_offline_args(server_args: str) -> str:
    """Derive the offline simple_inference args by dropping online-only flags."""
    tokens = shlex.split(server_args)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _OFFLINE_DROP_FLAGS_WITH_VALUE:
            i += 2  # skip flag + its value
            continue
        if tok in _OFFLINE_DROP_BOOL_FLAGS:
            i += 1
            continue
        out.append(tok)
        i += 1
    return shlex.join(out)


def _env_prefix(env_vars: str) -> str:
    """Newline-separated env_vars -> single-line `VAR=val VAR=val` prefix."""
    parts = [p.strip() for p in env_vars.replace("\\n", "\n").splitlines()]
    return " ".join(p for p in parts if p)


def resolve(entry: dict) -> dict:
    extra_args = _get(entry, _EXTRA_ARGS_KEYS)
    tp, server_extra_args = _split_tp(extra_args)
    client_command = entry.get("client_command", "") or ""
    return {
        "label": entry["model_name"],
        "model_path": entry["model_path"],
        "test_level": entry.get("test_level", ""),
        "tp": tp if tp is not None else 1,
        "tp_from_catalog": tp is not None,
        "env_prefix": _env_prefix(str(entry.get("env_vars", ""))),
        "server_extra_args": server_extra_args,
        "simple_inference_args": _to_offline_args(server_extra_args),
        "is_mtp": "--method mtp" in extra_args or "--method eagle" in extra_args,
        "fewshot": int(entry.get("lm_eval_num_fewshot", DEFAULT_FEWSHOT)),
        "num_concurrent": int(
            entry.get("lm_eval_num_concurrent", DEFAULT_NUM_CONCURRENT)
        ),
        "client_command": client_command,
        "threshold": _get(entry, _THRESHOLD_KEYS, default=None),
        "baseline": entry.get("accuracy_baseline"),
        "baseline_note": entry.get("_baseline_note", ""),
    }


def match(entry: dict, model_filter: str) -> bool:
    f = model_filter.lower()
    return (
        f in entry.get("model_name", "").lower()
        or f in entry.get("model_path", "").lower()
    )


def repo_root() -> Path:
    """Locate the ATOM repo root.

    Prefer `git rev-parse --show-toplevel` from the CWD so the script works
    whether it lives at repo-root `scripts/` or is bundled inside the skill
    folder — the skill is always invoked from the repo root. Fall back to the
    file-relative guess (scripts/ is a direct child of the repo root)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        return Path(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return Path(__file__).resolve().parent.parent


def default_catalog() -> Path:
    return repo_root() / CATALOG_REL_PATH


def load_local_catalog(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(f"catalog not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_remote_catalog(ref: str) -> list:
    """Read the catalog from a git ref (e.g. origin/main) instead of the working
    tree. Fetches the ref first (best-effort) so the result is the latest CI
    matrix even when the local branch is behind."""
    root = repo_root()
    # Best-effort fetch of the remote so origin/<branch> is up to date. A ref like
    # "origin/main" -> `git fetch origin main`; a bare tag/branch is left as-is.
    if "/" in ref:
        remote, _, branch = ref.partition("/")
        try:
            subprocess.run(
                ["git", "-C", str(root), "fetch", "--quiet", remote, branch],
                check=True,
                capture_output=True,
                timeout=60,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(
                f"WARNING: `git fetch {remote} {branch}` failed ({e}); "
                f"reading whatever {ref} points to locally.",
                file=sys.stderr,
            )
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "show", f"{ref}:{CATALOG_REL_PATH}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"could not read {CATALOG_REL_PATH} from ref '{ref}': "
            f"{e.stderr.strip() or e}"
        ) from e
    return json.loads(out.stdout)


def render_table(plan: list[dict]) -> str:
    lines: list[str] = []
    lines.append(f"Resolved {len(plan)} accuracy config(s):\n")
    for i, c in enumerate(plan, 1):
        tp_note = "" if c["tp_from_catalog"] else "  (defaulted — no -tp in catalog)"
        lines.append(f"[{i}] {c['label']}   (test_level={c['test_level']})")
        lines.append(f"    model_path : {c['model_path']}")
        lines.append(f"    TP         : {c['tp']}{tp_note}")
        if c["env_prefix"]:
            lines.append(f"    env        : {c['env_prefix']}")
        lines.append(f"    server args: {c['server_extra_args']}")
        lines.append(f"    offline    : {c['simple_inference_args']}")
        if c["client_command"]:
            lines.append(f"    client_cmd : (custom) {c['client_command'][:80]}...")
        else:
            lines.append(
                f"    gsm8k      : fewshot={c['fewshot']} concurrent={c['num_concurrent']}"
            )
        thr = c["threshold"]
        base = c["baseline"]
        lines.append(f"    threshold  : {thr}   baseline: {base}")
        if c["is_mtp"]:
            lines.append(
                "    NOTE: MTP variant — gsm8k accuracy alone cannot guard the "
                "draft head; MTP acceptance must be checked from /metrics."
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_filter", help="Substring matched vs model_name/model_path")
    ap.add_argument(
        "--test-level",
        default="",
        help="Comma list of levels to include (pr,nightly,main). Default: all.",
    )
    ap.add_argument("--format", choices=("table", "json"), default="table")
    src = ap.add_mutually_exclusive_group()
    src.add_argument(
        "--catalog", default=None, help="Path to a local models_accuracy.json"
    )
    src.add_argument(
        "--from-remote",
        nargs="?",
        const=DEFAULT_REMOTE_REF,
        default=None,
        metavar="REF",
        help=(
            "Read the catalog from a git ref (fetched first) instead of the "
            f"working tree. REF defaults to {DEFAULT_REMOTE_REF}."
        ),
    )
    args = ap.parse_args(argv)

    try:
        if args.from_remote is not None:
            catalog = load_remote_catalog(args.from_remote)
            source = f"{args.from_remote}:{CATALOG_REL_PATH}"
        else:
            catalog_path = Path(args.catalog) if args.catalog else default_catalog()
            catalog = load_local_catalog(catalog_path)
            source = str(catalog_path)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    levels = {x.strip() for x in args.test_level.split(",") if x.strip()}

    matched = [
        e
        for e in catalog
        if match(e, args.model_filter) and (not levels or e.get("test_level") in levels)
    ]
    if not matched:
        print(
            f"No catalog entries match model='{args.model_filter}'"
            + (f" test_level in {sorted(levels)}" if levels else "")
            + f"\nCatalog: {source}",
            file=sys.stderr,
        )
        return 1

    plan = [resolve(e) for e in matched]

    if args.format == "json":
        print(json.dumps(plan, indent=2, ensure_ascii=False))
    else:
        print(render_table(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
