---
name: git-workflow
description: Use when the task involves Git operations for the public xLLM repository, including choosing branch or tag names, preparing commits and pull requests, backporting fixes, checking repo-specific review expectations, or drafting commit messages from actual diffs.
---

# Git Workflow

Use xLLM repo reality, not generic Git habits.

## Reference Map

Load only the file that matches the user's immediate Git task.

| File | What it is for | When to load it |
| --- | --- | --- |
| `references/source-of-truth.md` | Repo-specific source priority and canonical files to consult | Load first when repo docs, local state, and user wording may disagree |
| `references/branch-and-pr.md` | Branch naming, day-to-day development flow, PR targeting, and review expectations | Load when the user asks about branch names, PR flow, reviewer expectations, or normal contribution steps |
| `references/release-and-backport.md` | Release branch naming, tag shapes, and backport flow | Load when the task mentions release branches, patch releases, cherry-picks, or hotfix/backport strategy |
| `references/commit-format.md` | Commit title/body conventions and xLLM-style examples | Load when the user asks for a commit message, commit style guidance, or message cleanup |

## Workflow

1. Decide which subtask the user actually needs.
2. Read `references/source-of-truth.md` when you need repo-specific confirmation.
3. Then load only the most relevant task file from the table above.
4. For commit message drafting, run `bash scripts/collect_git_context.sh [--staged|--all|--unstaged]` before writing the final message.
5. Draft commit messages from the actual diff, not from filenames alone.
6. If one diff mixes unrelated concerns, recommend splitting the commit instead of forcing one vague summary.
7. Default PR targets to `main` unless the task is clearly a release or backport flow.
8. For released lines, prefer landing on `main` first and then backporting unless the user explicitly wants a direct hotfix flow.

## Output

Return the smallest useful answer for the user's Git task:

- workflow questions: concrete branch, tag, PR, or backport steps
- commit message requests: `<type>: <subject>` plus an optional short bullet body
- repo-convention questions: current xLLM-specific guidance, not generic Git advice

## Quick Checks

- branch names match xLLM style such as `feat/<topic>`, `bugfix/<topic>`, or `release/vX.Y.Z`
- PR target is `main` unless this is a release or backport task
- commit title matches the dominant change in the diff
- release tags use semantic versions such as `v0.9.0` or `v0.9.1`
- owner-review expectations come from `.github/CODEOWNERS` when relevant
