---
name: git-commit-message
description: Generate a concise git commit message from staged or local repository changes. Use when Codex needs to inspect a diff, summarize the dominant intent, and draft a commit title and optional body before `git commit`.
---

# Git Commit Message

Draft the message from the actual diff, not from filenames alone.

## Workflow

1. Run `bash scripts/collect_git_context.sh`.
2. Use `--staged` when the next commit should describe only indexed changes.
3. Use `--all` when nothing is staged or the user explicitly wants one summary for every local change.
4. Read `references/commit-format.md` before drafting the final message.
5. If the diff mixes unrelated concerns, say so and recommend splitting the commit instead of forcing one vague message.

## Output

Return:

```text
<type>: <subject>

- optional detail
- optional detail
```

Omit the body when the title is already clear.

## Writing Rules

- Use exactly one lowercase type from `feat|bugfix|docs|test|refactor|chore|style|revert|perf|model|build`.
- Format the first line as `<type>: <subject>`.
- Keep the type and subject fully lowercase unless a path or product name requires another case.
- Write a subject with at least 4 words.
- End the subject with a period.
- Keep the subject concise even with the 4-word minimum.
- Describe the effect or intent, not a mechanical list of touched files.
- Prefer the smallest accurate type from `references/commit-format.md`.
- Mention breaking behavior, migrations, or follow-up work in the body when needed.
- If confidence is low because the diff is partial or noisy, say that explicitly.

## Quick Checks

- Confirm whether the generated message matches staged changes or all local changes.
- Check that the first line starts with a valid lowercase type, a colon, and one space.
- Check that the subject has at least 4 words and ends with a period.
- Check that the type, subject, and body all point to the same primary change.
- Rewrite broad subjects like `update code` into something specific and reviewable.
