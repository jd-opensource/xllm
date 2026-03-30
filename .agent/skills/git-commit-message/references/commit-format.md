# Commit Format

Use this exact first-line format:

```text
<type>: <subject>
```

Allowed lowercase types:

- `feat`: add user-visible behavior or a new capability
- `bugfix`: correct incorrect behavior or a regression
- `docs`: change documentation only
- `test`: add or update tests only
- `refactor`: improve structure without changing intended behavior
- `chore`: repository maintenance that does not fit the other types
- `style`: formatting or style-only changes without logic changes
- `revert`: revert an earlier commit
- `perf`: improve runtime or memory behavior
- `model`: change model definitions, checkpoints, prompts, or inference behavior
- `build`: change build, release, or dependency wiring

Subject guidelines:

- Use lowercase letters by default
- Include at least 4 words
- End with a period
- Start with a verb like `add`, `fix`, `remove`, `refactor`, `document`
- Keep it specific enough that a reviewer understands the main change
- Avoid filler like `update`, `misc`, `stuff`, `changes`

Body guidelines:

- Add a body only when the title alone is not enough
- Use short bullets for secondary details or important context
- Mention follow-up work, migration steps, or compatibility impact when relevant
