# submit_pr — examples

Worked examples showing how the same skill adapts to different repos. The point
is that the commit and PR always look like they belong in *that* repo.

## Example 1 — repo using Conventional Commits

`git log --oneline` shows: `feat(kv): ...`, `fix(mla): ...`, `refactor: ...`

**Change:** added an fp8 prefill path to the sparse attention kernel.

Commit subject:
```
feat(attn): add fp8 sparse prefill path
```

PR title: `feat(attn): add fp8 sparse prefill path`

PR body:
```markdown
## Summary
Adds an fp8 code path to the sparse-attention prefill kernel, gated behind the
existing `use_fp8` flag so the bf16 path is unchanged by default.

## Changes
- New `pa_sparse_prefill_fp8` entry point wired into the dispatch.
- Flag plumbing from the Python layer down to the kernel launcher.

## Test plan
- `pytest tests/test_sparse_prefill.py -k fp8` — 12 passed.
- Numerical parity vs. bf16 within tolerance on the smoke config.
```

## Example 2 — repo using plain imperative subjects

`git log --oneline` shows: `Add ...`, `Fix ...`, `Update ...` (no `type(scope):`)

**Change:** one-line fix for a wrong argument in a decode call.

Commit subject:
```
Fix missing page_size arg in V4 asm decode call
```

PR title: `Fix missing page_size arg in V4 asm decode call`

PR body (kept short because the change is small):
```markdown
## Summary
The V4 asm decode call site was missing `kv_last_page_lens` and `page_size`,
causing an incorrect merge. This passes them through.

## Test plan
- Reran the decode smoke test — output now matches the reference.
```

## Example 3 — fork workflow, `gh` missing then installed

**Setup:** `origin` = your fork `amd-ruitang3/aiter`, `upstream` = `ROCm/aiter`,
upstream default branch `main`. `gh` is not on PATH at first.

Flow:
1. Branch `fp8-prefill-path` created off the synced base.
2. Commit written in upstream's style.
3. `git push -u origin fp8-prefill-path` → succeeds.
4. `gh` missing → run `scripts/ensure_gh.sh`, which installs it (e.g. via
   `sudo apt install gh`). `helper.py` gives the fork-aware target.
5. If `gh auth status` is fine, open the PR straight against upstream:
   ```bash
   gh pr create --repo ROCm/aiter --base main \
     --head amd-ruitang3:fp8-prefill-path --title "..." --body-file body.md
   ```
6. Only if `gh` could not be installed at all, fall back to the compare URL
   (derived by `helper.py`, so the owner/repo is correct):
   ```
   https://github.com/ROCm/aiter/compare/main...amd-ruitang3:fp8-prefill-path?expand=1
   ```

## Example 4 — "just make a PR", nothing else said

User says only: "ok make a PR for this".

The skill still runs the full investigation, infers everything (base branch,
style, branch name from the diff), drafts title + body, and stops at the phase-4
confirmation to show the plan — rather than guessing silently and pushing. The
brevity of the request doesn't change the checkpoint before outward-facing steps.
