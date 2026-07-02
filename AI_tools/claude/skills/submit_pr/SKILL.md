---
name: submit_pr
description: >-
  Create a GitHub pull request from the current git repo: inspect changes,
  craft a commit and PR that match the repo's own conventions, push a branch,
  and open the PR with `gh` (installing `gh` first if it's missing, and only
  falling back to a ready-to-use compare URL if it truly can't). Use this
  whenever the user wants to "submit a PR", "open a
  pull request", "raise a PR", "push my changes for review", "send this
  upstream", or otherwise turn local work into a reviewable PR — including
  fork-based workflows against an upstream repo. Trigger even when the user
  only says "make a PR" or "let's get this reviewed" without more detail.
---

# submit_pr

Turn the current local work into a clean GitHub pull request with as little
friction as possible, while matching the target repo's existing style rather
than imposing a foreign one.

The guiding idea: a good PR reads like it belongs in the repo. So before writing
anything, look at how this repo already writes commits and PRs, and follow that.
Never fabricate — every claim in the summary and test plan must be grounded in
the actual diff and in commands you actually ran.

## Workflow

Work through these phases in order. Do the read-only investigation up front and
in parallel, then pause for confirmation before anything that leaves the local
machine (push, PR creation).

### 1. Understand the current state (read-only, run in parallel)

Gather all of this before deciding anything:

```bash
git status                    # what's staged / unstaged / untracked
git branch --show-current     # are we on a feature branch or on main/master?
git remote -v                 # origin only, or fork + upstream?
git log --oneline -15         # recent commit-message style to mimic
git diff                      # unstaged changes
git diff --cached             # staged changes
```

To capture the full scope of what the PR will contain, diff against the base
branch, not just the working tree: `git diff <base>...HEAD` (see phase 3 for
picking `<base>`).

From this, determine:
- **Base branch**: the default branch (usually `main` or `master`). Check with
  `git remote show origin | sed -n 's/.*HEAD branch: //p'` or
  `git symbolic-ref refs/remotes/origin/HEAD`.
- **Fork vs. direct**: if there's an `upstream` remote distinct from `origin`,
  this is a fork workflow — the PR base is `upstream`'s default branch and the
  head is `origin/<your-branch>`.
- **Commit style**: read `git log` — is it Conventional Commits
  (`feat(scope): ...`), plain imperative (`Add ...`), or something else? Match it.

### 2. Enforce the required identity and formatting (hard gates)

These are non-negotiable for this user's PRs. Do them before committing — a PR
that violates any of these should not be created.

**Git identity.** Every commit must be authored as this user. Verify and set it
on the local repo (not `--global`, so other repos are untouched):

```bash
git config user.name  amd-ruitang3
git config user.email Rui.Tang2@amd.com
```

Setting these is idempotent — run them so the commit is guaranteed correct
regardless of what the repo or global config held. If commits were already made
on the branch with the wrong identity, flag it (they may need
`git commit --amend --reset-author` or a rebase) rather than silently proceeding.

**Python formatting.** Every modified/added `.py` file that will be in the PR
must pass both `black` and `ruff format`. Format them before staging:

```bash
# collect the python files in the change (tracked + staged + untracked)
files=$(git diff --name-only --diff-filter=d; \
        git diff --cached --name-only --diff-filter=d; \
        git ls-files --others --exclude-standard \
       | sort -u | grep -E '\.py$')

# format them (both, in this order); ruff format is idempotent after black
[ -n "$files" ] && black $files && ruff format $files
```

If `black` or `ruff` isn't installed, don't skip the gate — tell the user and
suggest `pip install black ruff` (or the repo's own dev-tooling command). After
formatting, re-run `git diff` so the formatting changes are included in the
commit. Formatting must happen *before* the commit in phase 3, so the committed
code is already clean — never commit first and reformat after.

### 3. Prepare the branch

- If currently on the base branch (`main`/`master`), create a feature branch —
  never commit review work directly onto the base branch. Name it descriptively
  from the change (e.g. `fp8-prefill-path`, `fix-mla-merge-sig`). Ask the user
  if you're unsure of a good name.
- If already on a suitable feature branch, stay on it.

### 4. Stage and commit

- Review what should go in. Don't blindly `git add -A` if there are unrelated
  stray files, build artifacts, or debug edits — mention anything that looks
  like it shouldn't be committed and let the user decide.
- Write the commit message in the repo's style (from phase 1). Keep the subject
  focused; add a body when the change needs a "why".
- End the commit message with the trailer:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  ```

### 5. Confirm before pushing

This is the checkpoint. Before pushing or creating the PR, show the user:
- the branch name and where it will push (`origin`),
- the base it will target,
- the commit subject(s),
- the proposed PR title and body,
- confirmation the hard gates passed: identity is `amd-ruitang3 <Rui.Tang2@amd.com>`
  and modified `.py` files were run through `black` + `ruff format`.

Then pause for a go-ahead. Pushing and opening a PR are outward-facing and hard
to fully undo, so don't do them until the user confirms — unless they've already
said "just do it / don't ask".

### 6. Push and open the PR

First push the branch:

```bash
git push -u origin <branch>
```

**Always derive the PR target with the bundled helper — do not hand-roll it.**
`scripts/helper.py` reads the *stored* remote URLs (not `git remote get-url`,
which can be rewritten by `url.*.insteadOf` and yield a bogus owner/repo),
detects fork vs. direct, and returns the base branch, base repo, fork-aware head
ref, and the exact compare URL as JSON:

```bash
python3 <skill-dir>/scripts/helper.py --branch <branch>
# -> { base_branch, base_repo, is_fork, head_ref, compare_url, ... }
```

Use those fields for both the `gh` path and the fallback — this is what keeps the
owner/repo correct, especially for forks.

**Ensure `gh` is available, installing it if needed.** Prefer opening the PR with
`gh`; only fall back to the compare URL if `gh` genuinely can't be made to work.
Run the bundled installer, which tries conda/brew/apt/dnf/prebuilt-binary and
prints the resolved path (exit 0) or fails (exit 1):

```bash
bash <skill-dir>/scripts/ensure_gh.sh && command -v gh
```

Then:

1. **`gh` present and authenticated** (`gh auth status` exits 0) — create the PR:
   ```bash
   gh pr create --base <base_branch> --head <branch> \
     --title "<title>" --body-file <tmp-body-file>
   # fork workflow: add --repo <base_repo>   (e.g. ROCm/aiter)
   # draft:         add --draft
   ```
   Use `--body-file` for multi-line bodies. On success, print the PR URL `gh`
   returns.

2. **`gh` present but NOT authenticated** — don't try to script the login. Tell
   the user to run it interactively (suggest they type `! gh auth login` in the
   prompt so it runs in-session). Meanwhile the branch is already pushed, so also
   give them the `compare_url` so they aren't blocked while auth is pending.

3. **`gh` could not be installed** — fall back gracefully. The push already
   succeeded, so print the `compare_url` from `helper.py` for the user to click:
   ```
   https://github.com/<base_repo>/compare/<base_branch>...<head_ref>?expand=1
   ```
   (For a fork, `helper.py` has already put the upstream repo in `base_repo` and
   `fork-owner:branch` in `head_ref`.)

See `reference.md` for the underlying derivation logic and `gh` edge cases.

## PR body structure

Default to a lean, honest body. Use this template, dropping sections that don't
apply:

```markdown
## Summary
<1-3 sentences: what changed and why, grounded in the diff>

## Changes
- <notable change 1>
- <notable change 2>

## Test plan
- <how it was verified — only commands actually run; if not tested, say so>
```

Only add a **Motivation**, **Perf / numerical notes**, or **Related issues**
(`Closes #123`) section when the change actually warrants it. A two-line fix
should get a two-line PR — don't pad it. Never claim tests passed that weren't
run; if verification is pending, write that plainly.

See `examples.md` for worked examples across different repo styles.

## Notes

- **Hard gates (never skip):** commit identity must be
  `amd-ruitang3 <Rui.Tang2@amd.com>` (set per-repo in phase 2), and every
  modified `.py` file must pass `black` and `ruff format` before the commit.
  A PR that violates either shouldn't go out.
- Match, don't impose. The single most common way to get this wrong is writing a
  Conventional-Commits message into a repo that uses plain imperative subjects,
  or vice versa. Always check `git log` first.
- Keep the working directory honest: report failures and skipped steps as they
  are. If `git push` is rejected or `gh` errors, surface the real output.
- Read-only git commands can run without asking. Anything that mutates remote
  state waits for the phase-4 confirmation.
