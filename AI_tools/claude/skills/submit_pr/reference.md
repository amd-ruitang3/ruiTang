# submit_pr — reference

Details you only need occasionally. Read this when the fork/URL derivation or a
`gh` edge case comes up.

## Deriving owner/repo from a remote

**Use `scripts/helper.py`.** It handles all of the below — fork detection, base
branch, fork-aware head ref, and the final compare URL — in one call, so you
don't have to reassemble it by hand (and get it wrong). This is the recommended
path; the manual details here are just so you understand what it does.

`git remote -v` prints URLs in one of two forms:

- SSH:   `git@github.com:OWNER/REPO.git`
- HTTPS: `https://github.com/OWNER/REPO.git`

To get the compare URL you need `OWNER/REPO` (strip the trailing `.git`).
Read the **stored** URL, not `git remote get-url` — the latter applies
`url.<base>.insteadOf` rewrites, which on some setups turn a github.com URL into
a local mirror path and give you a garbage owner/repo:

```bash
git config --get remote.origin.url \
  | sed -E 's#(git@|https://|ssh://git@)##; s#github\.com[:/]##; s#\.git$##'
# -> OWNER/REPO
```

Do the same with `upstream` when a fork workflow is in play. Again: prefer
`helper.py` so this is done consistently.

## Installing gh

`scripts/ensure_gh.sh` tries, in order: existing `gh`, conda/mamba, brew, passwordless
`sudo apt`/`dnf`, then a prebuilt release binary into `~/.local/bin`. It prints
the resolved `gh` path and exits 0 on success, or exits 1 so the caller knows to
use the compare-URL fallback. If it installed into `~/.local/bin`, make sure that
dir is on `PATH` for the current shell.

## Picking the base branch

In order of reliability:

```bash
# 1. Ask the remote directly (most reliable, needs network):
git remote show origin | sed -n 's/.*HEAD branch: //p'

# 2. From the cached symbolic ref:
git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's#refs/remotes/origin/##'

# 3. Fall back to whichever of these exists:
git rev-parse --verify origin/main   >/dev/null 2>&1 && echo main
git rev-parse --verify origin/master >/dev/null 2>&1 && echo master
```

For a fork, the PR base is the **upstream** default branch, found the same way
against the `upstream` remote.

## Fork workflow specifics

A fork setup looks like:

```
origin    git@github.com:YOU/repo.git       (your fork — you push here)
upstream  git@github.com:PROJECT/repo.git   (the real repo — PR targets here)
```

- Push your branch to `origin`.
- With `gh`: `gh pr create --repo PROJECT/repo --base <upstream-default> --head YOU:<branch> ...`
- Fallback compare URL:
  `https://github.com/PROJECT/repo/compare/<upstream-default>...YOU:<branch>?expand=1`

Keep your fork's base branch synced before branching when possible
(`git fetch upstream && git merge --ff-only upstream/<base>`), so the PR diff is
clean and doesn't include unrelated upstream drift.

## gh CLI notes

- Check presence: `command -v gh`
- Check auth: `gh auth status` (exit 0 = authenticated). If not authenticated,
  don't try to script the login — tell the user to run `gh auth login` themselves
  (suggest they type `! gh auth login` in the prompt so the interactive flow runs
  in-session), then continue.
- Useful flags: `--draft` (open as draft), `--web` (open the created PR in a
  browser), `--reviewer USER`, `--label NAME`, `--assignee @me`.
- To pass a multi-line body safely, write it to a temp file and use
  `--body-file <file>` instead of cramming newlines into `--body`.

## When there is no `gh` and no network for it

Everything up to and including `git push -u origin <branch>` still works offline
to the extent the remote is reachable. Once the branch is pushed, the compare URL
is all the user needs to finish opening the PR by hand — always print it as the
final output in the fallback path.
