#!/usr/bin/env python3
"""submit_pr helper — gather PR context and build the fallback compare URL.

Run from inside the target git repo. This does read-only git queries and prints
a small JSON blob the skill can use to fill in the base branch, detect a fork
workflow, and construct the GitHub compare URL for the no-`gh` fallback path.

Usage:
    python helper.py [--branch <name>]

If --branch is omitted the current branch is used.
"""

import argparse
import json
import re
import subprocess
import sys


def git(*args):
    """Run a read-only git command, returning stripped stdout ('' on failure)."""
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except subprocess.CalledProcessError:
        return ""


def owner_repo(remote):
    """Return 'OWNER/REPO' for a remote, or '' if the remote is absent.

    Read the stored config value rather than `git remote get-url`, because the
    latter applies any `url.<base>.insteadOf` rewrites — which can turn a
    github.com URL into a local mirror path and break owner/repo derivation.
    """
    url = git("config", "--get", f"remote.{remote}.url")
    if not url:
        url = git("remote", "get-url", remote)
    if not url:
        return ""
    url = re.sub(r"^(git@|https://|ssh://git@)", "", url)
    url = re.sub(r"^github\.com[:/]", "", url)
    url = re.sub(r"\.git$", "", url)
    return url


def default_branch(remote):
    """Best-effort default branch for a remote."""
    ref = git("symbolic-ref", f"refs/remotes/{remote}/HEAD")
    if ref:
        return ref.rsplit("/", 1)[-1]
    for cand in ("main", "master"):
        if git("rev-parse", "--verify", f"{remote}/{cand}"):
            return cand
    return "main"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default=None)
    args = ap.parse_args()

    if not git("rev-parse", "--is-inside-work-tree"):
        print("error: not inside a git work tree", file=sys.stderr)
        return 1

    branch = args.branch or git("branch", "--show-current")
    remotes = git("remote").splitlines()

    origin = owner_repo("origin")
    upstream = owner_repo("upstream") if "upstream" in remotes else ""
    is_fork = bool(upstream) and upstream != origin

    if is_fork:
        base_remote, base_repo = "upstream", upstream
        fork_owner = origin.split("/")[0] if origin else ""
        head_ref = f"{fork_owner}:{branch}" if fork_owner else branch
    else:
        base_remote, base_repo = "origin", origin
        head_ref = branch

    base = default_branch(base_remote)
    compare_url = (
        f"https://github.com/{base_repo}/compare/{base}...{head_ref}?expand=1"
        if base_repo
        else ""
    )

    print(
        json.dumps(
            {
                "branch": branch,
                "remotes": remotes,
                "origin": origin,
                "upstream": upstream,
                "is_fork": is_fork,
                "base_branch": base,
                "base_repo": base_repo,
                "head_ref": head_ref,
                "compare_url": compare_url,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
