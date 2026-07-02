#!/usr/bin/env bash
# Ensure the GitHub CLI (`gh`) is available. Print the usable gh path on stdout
# and exit 0 if gh is present or was installed; exit 1 if every method failed
# (the caller should then use the compare-URL fallback).
#
# Tries, in order:
#   1. gh already on PATH
#   2. conda / mamba  (conda-forge has gh; common on ML boxes, no sudo)
#   3. brew           (Linuxbrew or macOS, no sudo)
#   4. sudo apt/dnf   (only if sudo is passwordless; fast + clean when it is)
#   5. prebuilt release binary -> ~/.local/bin  (no sudo, needs network to github)
#
# All progress chatter goes to stderr so stdout is just the final gh path.
set -uo pipefail
say() { echo "[ensure_gh] $*" >&2; }

if command -v gh >/dev/null 2>&1; then
  say "gh already installed: $(gh --version 2>&1 | head -1)"
  command -v gh
  exit 0
fi

say "gh not found — attempting installation…"

try_conda() {
  local mgr
  for mgr in mamba conda; do
    if command -v "$mgr" >/dev/null 2>&1; then
      say "trying $mgr install -c conda-forge gh"
      "$mgr" install -y -c conda-forge gh >&2 2>&1 && command -v gh >/dev/null 2>&1 && return 0
    fi
  done
  return 1
}

try_brew() {
  command -v brew >/dev/null 2>&1 || return 1
  say "trying brew install gh"
  brew install gh >&2 2>&1 && command -v gh >/dev/null 2>&1
}

try_binary() {
  command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1 || return 1
  local os arch tag ver url tmp dl
  os=$(uname -s | tr '[:upper:]' '[:lower:]')   # linux / darwin
  case "$(uname -m)" in
    x86_64|amd64)  arch=amd64 ;;
    aarch64|arm64) arch=arm64 ;;
    *) say "unknown arch $(uname -m)"; return 1 ;;
  esac

  fetch() { if command -v curl >/dev/null 2>&1; then curl -fsSL "$1"; else wget -qO- "$1"; fi; }

  say "resolving latest gh release tag"
  tag=$(fetch https://api.github.com/repos/cli/cli/releases/latest \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])' 2>/dev/null)
  [ -n "${tag:-}" ] || { say "could not resolve latest tag"; return 1; }
  ver=${tag#v}
  url="https://github.com/cli/cli/releases/download/${tag}/gh_${ver}_${os}_${arch}.tar.gz"

  tmp=$(mktemp -d)
  dl="$tmp/gh.tgz"
  say "downloading $url"
  if command -v curl >/dev/null 2>&1; then curl -fsSL "$url" -o "$dl"; else wget -qO "$dl" "$url"; fi
  [ -s "$dl" ] || { say "download failed"; rm -rf "$tmp"; return 1; }

  tar -xzf "$dl" -C "$tmp" || { rm -rf "$tmp"; return 1; }
  mkdir -p "$HOME/.local/bin"
  cp "$tmp"/gh_*/bin/gh "$HOME/.local/bin/gh" || { rm -rf "$tmp"; return 1; }
  chmod +x "$HOME/.local/bin/gh"
  rm -rf "$tmp"
  export PATH="$HOME/.local/bin:$PATH"
  say "installed to ~/.local/bin/gh (ensure ~/.local/bin is on PATH)"
  command -v gh >/dev/null 2>&1
}

try_sudo_pkg() {
  command -v sudo >/dev/null 2>&1 || return 1
  sudo -n true 2>/dev/null || { say "sudo needs a password — skipping apt/dnf"; return 1; }
  if command -v apt-get >/dev/null 2>&1; then
    say "trying sudo apt-get install gh"
    sudo -n apt-get update >&2 2>&1 && sudo -n apt-get install -y gh >&2 2>&1 && command -v gh >/dev/null 2>&1 && return 0
  fi
  if command -v dnf >/dev/null 2>&1; then
    say "trying sudo dnf install gh"
    sudo -n dnf install -y gh >&2 2>&1 && command -v gh >/dev/null 2>&1 && return 0
  fi
  return 1
}

for m in try_conda try_brew try_sudo_pkg try_binary; do
  if $m; then
    say "success via $m: $(gh --version 2>&1 | head -1)"
    command -v gh
    exit 0
  fi
done

say "all installation methods failed — caller should use the compare-URL fallback"
exit 1
