# Commit Format

Use this exact first-line format:

```text
<type>: <subject>
```

Allowed lowercase types:

- `feat`: add user-visible behavior or a new capability.
- `bugfix`: correct incorrect behavior or a regression.
- `docs`: change documentation only.
- `test`: add or update tests only.
- `refactor`: improve structure without changing intended behavior.
- `chore`: repository maintenance that does not fit the other types.
- `style`: formatting or style-only changes without logic changes.
- `revert`: revert an earlier commit.
- `perf`: improve runtime or memory behavior.
- `model`: change model definitions, checkpoints, prompts, or inference behavior.
- `build`: change build, release, or dependency wiring.
- `release`: change release versioning, release notes, or release-only metadata.

Subject guidelines:

- use lowercase letters by default
- include at least 4 words
- end with a period
- start with a verb like `add`, `fix`, `remove`, `refactor`, `document`
- keep it specific enough that a reviewer understands the main change
- avoid filler like `update`, `misc`, `stuff`, `changes`
- describe the effect or intent, not a mechanical file list

Body guidelines:

- add a body only when the title alone is not enough
- use short bullets for secondary details or important context
- mention follow-up work, migration steps, or compatibility impact when relevant
- if confidence is low because the diff is partial or noisy, say that explicitly

Notes for xLLM:

- `bugfix:` appears more often than `fix:` in current history and should be the default bug-repair prefix
- `fix:` exists in a few historical commits but is less consistent than `bugfix:`
- PR-number suffixes like `(#1142)` are common in merged history but are optional unless the user explicitly wants them
- staged series markers like `[1/N]` or `[3/N]` appear when a change is intentionally split across multiple commits
