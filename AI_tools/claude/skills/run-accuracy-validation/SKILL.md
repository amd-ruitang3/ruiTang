---
name: run-accuracy-validation
description: Run the FULL set of CI accuracy tests for one model — given a model path, discover every matching permutation in the CI accuracy catalog (.github/benchmark/models_accuracy.json) and run them all: an offline simple_inference smoke plus every gsm8k (lm_eval) config across all test_levels (pr/nightly/main), then compare each result against its CI threshold/baseline. Use whenever the user says "测一下 DeepSeek-V4 的精度", "test model X accuracy", "跑完整精度", "把 CI 里所有相关的精度跑一遍", "run all accuracy tests for <model>", "验证 <model> 精度有没有回归", or names a model + accuracy/精度/gsm8k without listing specific flags — the whole point is that the flag permutations come from the catalog, not the user. For a SINGLE known config (one specific fewshot/mtp run the user fully specifies), use run-atom-workload directly instead.
version: 1.0.0
scope: ATOM on AMD ROCm; wraps run-atom-workload; reads .github/benchmark/models_accuracy.json
last_updated: 2026-07-13
---

## Install / dependencies

This skill runs *inside* an ATOM checkout — it needs:
- the **run-atom-workload** scripts at repo-root `scripts/` (stop/start/drain/…),
- the CI catalog at `.github/benchmark/models_accuracy.json`,
- a GPU box that can serve the model.

The resolver `resolve_accuracy_configs.py` is **bundled with this skill** (in its
own `scripts/`). Invoke it by its full path so it works regardless of CWD:

```
python3 ~/.claude/skills/run-accuracy-validation/scripts/resolve_accuracy_configs.py ...
```

It locates the ATOM repo and catalog via `git rev-parse --show-toplevel`, so run
it with the CWD inside the target ATOM checkout. The run-atom-workload scripts
(stop/start/drain/…) are the ATOM repo's own repo-root `scripts/*` and are
referenced repo-root-relative below.

## Proxy on this box (READ FIRST)

This container **sometimes has an HTTP proxy set** (`HTTP_PROXY`/`HTTPS_PROXY`,
e.g. `http://10.2.80.7:18124`) with no `no_proxy`. When set, every localhost
call routes through the proxy and fails instantly — the `/v1/models` ready-poll,
the lm_eval / gsm8k client, simple_inference — which looks like a server crash
but is pure client networking. A manual `curl` may still work and mask it.

**Prefix every server-start AND every client command with:**

```
no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1
```

Below this is written as `<NOPROXY>`. It's surgical — localhost bypasses the
proxy while HF downloads still work. (If nothing needs external fetch, fully
`unset HTTP_PROXY HTTPS_PROXY` is equivalent.) Do this whether or not you *think*
the proxy is set — it's a no-op when it isn't, and the failure mode is silent
and expensive when it is.

## Why this skill exists

CI validates a model's accuracy across a *matrix* of configs — base, MTP,
DP-attention+TBO, online-quant, different quant checkpoints — each with its own
env vars, server flags, and pass/fail threshold. When someone says "test
DeepSeek-V4 accuracy" they mean **all of them**, not one hand-picked run. Getting
that matrix right by memory is error-prone: the flags, the `-tp` value, the
model-family env vars, and the thresholds all live in
`.github/benchmark/models_accuracy.json` and drift over time.

This skill makes the catalog the source of truth: it resolves the matrix for a
model, then drives each config through the proven **run-atom-workload** 5-step
flow. It does not reimplement server orchestration — that lives in
[[run-atom-workload]] and its `scripts/*`. This skill is the layer on top that
answers "*which* runs, with *what* args, judged against *what* threshold".

## Inputs

The user supplies a **model path** (usually a local path like
`/data/DeepSeek-V4-Pro`). They may also narrow scope ("just the pr ones",
"只跑 nightly"). Everything else — the permutations — comes from the catalog.

Two things to keep distinct:
- **Catalog `model_path`** is an HF id (`deepseek-ai/DeepSeek-V4-Pro`). It is
  used only to *match* catalog entries and as the lm_eval `model=` name.
- **The path you actually serve** is what the user gave you. Substitute the
  user's local path into the run commands; use the catalog only for the
  args/env/threshold of each permutation.

## Step A — resolve the run plan (always first)

Derive a filter from the model (the basename usually works: `V4`,
`DeepSeek-V4-Pro`, `R1-0528`) and list the matrix:

```bash
python3 ~/.claude/skills/run-accuracy-validation/scripts/resolve_accuracy_configs.py <FILTER> --format table
```

- Omit `--test-level` to get **all levels** (the default — "run all related").
- Add `--test-level pr` / `pr,nightly` when the user scoped it.
- `--format json` gives the same plan machine-readably (fields below) when you
  want to drive the loop programmatically.
- The catalog read is the **live CI source of truth**, not hardcoded — the
  script reads it fresh each run, so any config CI adds/changes/removes flows
  through automatically. By default it reads your locally checked-out copy
  (freshness follows your git state). To force the latest regardless of your
  branch, add `--from-remote` (fetches and reads `origin/main`) or
  `--from-remote <ref>`. If the user says "跑最新的" / "对齐 CI 最新", use it.

Each resolved config carries: `label`, `model_path`, `test_level`, `tp`,
`env_prefix`, `server_extra_args`, `simple_inference_args`, `is_mtp`, `fewshot`,
`num_concurrent`, `client_command` (empty = default gsm8k), `threshold`,
`baseline`.

**Show the resolved plan to the user and confirm before running** — this is
where they catch "actually skip the nightly one" or "wrong model". Running the
full matrix is expensive (each config is a full server cold-start + a
1319-sample gsm8k, ~30 min for a large MoE), so the confirmation earns its keep.

If the filter matches nothing, the script exits non-zero — widen the filter or
check the model name against the catalog; do not guess flags.

## Step B — offline simple_inference smoke (once, first)

Before the long gsm8k matrix, run one quick offline smoke so a
model-won't-even-load failure surfaces in minutes, not after a full server
spin-up. Use the **first** resolved config's `env_prefix` +
`simple_inference_args` (the offline-safe subset — serving-throughput flags like
`--enable-tbo` are already stripped by the resolver):

Follow [[run-atom-workload]]'s **offline** flow (steps 1 → 2+3 fused → 4 → 5):

```bash
# 1 clean
bash scripts/stop_atom_server.sh
# 2+3 offline workload (note trailing &) — NOPROXY + env_prefix + local path + TP + offline args
<NOPROXY> <ENV_PREFIX> AITER_LOG_LEVEL=WARNING bash scripts/start_simple_inference.sh <LOCAL_PATH> <TP> <SIMPLE_INFERENCE_ARGS> &
# 4 drain (offline mode; PORT unused)
bash scripts/wait_infer_drain.sh 0 15 10
# 5 teardown
bash scripts/stop_atom_server.sh
```

Read the result: `grep -E "^Generated|^Output|tokens/s" /app/logs_claude/simple_inference.log`.
Coherent output = smoke passed → proceed to the gsm8k matrix. A fault (drain
exit 2) or garbage output → stop and report; don't burn hours on the matrix if
the model can't load.

## Step C — run each gsm8k config through run-atom-workload

For **each** resolved config, run the server-based 5-step flow from
[[run-atom-workload]]. Run configs **sequentially** (they each need all GPUs);
never start a second server before `stop_atom_server.sh` from the previous one.

**Use PORT 8000.** The custom `client_command` entries in the catalog hardcode
`base_url=http://localhost:8000/...`, so serving on 8000 lets those commands run
verbatim. If you must use another port, you have to rewrite the `base_url` port
in every custom client_command to match — 8000 avoids that entirely.

Per config:

```bash
# 1 clean
bash scripts/stop_atom_server.sh

# 2 start server — NOPROXY + env_prefix + user's local path + TP + server_extra_args, trailing &
<NOPROXY> <ENV_PREFIX> AITER_LOG_LEVEL=WARNING bash scripts/start_atom_server.sh <LOCAL_PATH> <TP> 8000 <SERVER_EXTRA_ARGS> &

# 2.5 MANDATORY ready gate (Bash tool timeout >= 900000ms for large MoE)
<NOPROXY> bash scripts/wait_server_ready.sh 8000 15 5 /app/logs_claude/atom_server.log
#   non-zero exit -> skip to this config's step 5, record FAILED-TO-START, continue to next config

# 3 gsm8k client (trailing &).  DEFAULT config:
<NOPROXY> bash scripts/run_gsm8k_eval.sh <CATALOG_MODEL_PATH> 8000 <FEWSHOT> &
#   -> use the catalog HF model_path as the lm_eval model= name.
#   -> for configs with a custom client_command, see "Custom client_command" below instead.

# 4 drain
bash scripts/wait_infer_drain.sh 8000 30 10

# 5 teardown (ALWAYS, even on fault)
bash scripts/stop_atom_server.sh

# read this config's result before moving on
grep -E "flexible-extract|strict-match" /app/logs_claude/gsm8k_eval.log | head -2
```

Capture the flexible-extract number for the summary **immediately** after each
config — the log is overwritten by the next config's run.

### Custom client_command

~5 catalog entries (e.g. the DP-attention conc1000 variants, gpt-oss) carry a
`client_command` — a full lm_eval invocation, usually `local-chat-completions`
with `${MODEL_PATH}` / `${OUTPUT_PATH}` placeholders. For these, do NOT use
`run_gsm8k_eval.sh`; instead run the catalog's command as step 3 with the
placeholders expanded (`MODEL_PATH` = catalog HF id, `OUTPUT_PATH` =
`/app/logs_claude/gsm8k_eval_json`), still wrapped with trailing `&`, still
prefixed with `<NOPROXY>`, and still gated by steps 2.5 and 4. Then read
flexible-extract from the emitted results
JSON: `jq '.results.gsm8k["exact_match,flexible-extract"]' <OUTPUT_PATH>/*.json`.

### MTP configs

For `is_mtp` configs, gsm8k accuracy alone does **not** guard the draft head —
speculative decoding is lossless w.r.t. the target model, so a broken draft head
leaves accuracy unchanged and only craters acceptance rate. Note this in the
summary. Capturing acceptance requires scraping the server `/metrics` endpoint
during the run (CI does this); if you didn't capture it, say so rather than
implying MTP is fully validated.

## Step D — judge and summarize

For each config compare the measured flexible-extract against the catalog:
- **PASS** if `measured >= threshold`.
- Also report the delta vs `baseline`: within the ±noise band (baseline note
  often quotes ±0.00XX; ~0.006 for a full 1319-sample gsm8k) = no regression;
  measured well below baseline but above threshold = soft warning worth flagging.

Present a single table, e.g.:

```
Model: /data/DeepSeek-V4-Pro   (3 configs)

Config                          level    measured   threshold  baseline  verdict
DeepSeek-V4-Pro                 pr       0.9522     0.94       0.96      PASS (−0.008 vs base, within noise)
DeepSeek-V4-Pro MTP             pr       0.9515     0.94       0.96      PASS (accept-rate NOT captured)
DeepSeek-V4-Pro TBO+DPA c1000   nightly  0.9490     0.93       0.95      PASS
simple_inference smoke          —        —          —          —         PASS (coherent output)
```

Report failures loudly (below threshold, failed to start, faulted) with the
tail of the relevant log so the user can act.

## Hard rules

1. **The catalog is the source of truth for permutations.** Never invent flags,
   TP, env vars, or thresholds — resolve them. If the user's request implies a
   config not in the catalog, say so.
2. **Serve the user's local path; match/judge with the catalog entry.** Don't
   try to serve the HF id unless that's literally what the user gave you.
3. **One config at a time, full teardown between.** Each needs all GPUs; a
   lingering server from the previous config will OOM the next.
4. **Confirm the resolved plan before running the matrix.** It's long and
   expensive; the user should see what's about to run.
5. **Delegate orchestration to [[run-atom-workload]].** All the failure-mode
   wisdom (background with `&`, mandatory `wait_server_ready` gate, drain as the
   only hang detector, no `&&` chaining, no wrapper scripts) applies unchanged —
   do not reimplement or shortcut it here.
6. **Record each result before the next config overwrites the log.**
7. **Prefix every server-start and client command with `<NOPROXY>`** on this box
   (`no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1`). A stray proxy
   silently routes localhost through it and fails the ready-poll / lm_eval,
   looking like a server crash. Harmless when no proxy is set. See "Proxy on
   this box".

## Reference

- Resolver (bundled): `~/.claude/skills/run-accuracy-validation/scripts/resolve_accuracy_configs.py MODEL_FILTER [--test-level ...] [--format table|json] [--from-remote [REF]]`
- Catalog: `.github/benchmark/models_accuracy.json` (CI source of truth)
- Underlying run flow: [[run-atom-workload]]
- Baselines/thresholds also mirrored per-model in the catalog entries themselves.
