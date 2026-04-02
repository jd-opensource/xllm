#!/usr/bin/env bash

set -euo pipefail

mode="auto"

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [--staged|--all|--unstaged]" >&2
  exit 1
fi

if [[ $# -eq 1 ]]; then
  case "$1" in
    --staged)
      mode="staged"
      ;;
    --all)
      mode="all"
      ;;
    --unstaged)
      mode="unstaged"
      ;;
    *)
      echo "unknown option: $1" >&2
      echo "usage: $0 [--staged|--all|--unstaged]" >&2
      exit 1
      ;;
  esac
fi

if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "not inside a git repository" >&2
  exit 1
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

staged_count="$(git diff --cached --name-only | wc -l | tr -d ' ')"
untracked_files="$(git ls-files --others --exclude-standard)"

if [[ "$mode" == "auto" ]]; then
  if [[ "$staged_count" != "0" ]]; then
    mode="staged"
  else
    mode="all"
  fi
fi

case "$mode" in
  staged)
    status_cmd=(git diff --cached --name-status)
    stat_cmd=(git diff --cached --stat)
    patch_cmd=(git diff --cached --unified=1 --no-color)
    headline="staged changes"
    ;;
  unstaged)
    status_cmd=(git diff --name-status)
    stat_cmd=(git diff --stat)
    patch_cmd=(git diff --unified=1 --no-color)
    headline="unstaged changes"
    ;;
  all)
    headline="all local changes"
    ;;
  *)
    echo "invalid mode: $mode" >&2
    exit 1
    ;;
 esac

echo "repo: $repo_root"
echo "branch: $(git branch --show-current)"
echo "scope: $headline"
echo
echo "status:"
git status --short
echo

if [[ -n "$untracked_files" ]]; then
  echo "untracked files:"
  printf '%s\n' "$untracked_files"
  echo
fi

if [[ "$mode" == "all" ]]; then
  echo "changed files:"
  git diff HEAD --name-status
  echo
  echo "diff stat:"
  git diff HEAD --stat
  echo
  echo "patch excerpt:"
  git diff HEAD --unified=1 --no-color | sed -n '1,400p'
else
  echo "changed files:"
  "${status_cmd[@]}"
  echo
  echo "diff stat:"
  "${stat_cmd[@]}"
  echo
  echo "patch excerpt:"
  "${patch_cmd[@]}" | sed -n '1,400p'
fi
