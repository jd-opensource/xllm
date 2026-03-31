# Branch Naming

Use one of these branch shapes:

```text
<type>/<topic>
<namespace>/<type>/<topic>
preview/<topic>
release/vX.Y.Z
```

Recommended lowercase branch types:

- `feat`: new user-visible capability or feature work
- `bugfix`: incorrect behavior, regressions, or hot fixes
- `refactor`: structural changes without intended behavior changes
- `docs`: documentation-only work when a dedicated branch is useful
- `test`: test-only work when separated from product changes
- `perf`: runtime or memory improvements
- `chore`: repo maintenance that does not fit the other categories
- `build`: dependency, CI, packaging, or release tooling changes

Topic guidelines:

- use lowercase letters, numbers, and hyphens by default
- keep the topic short, specific, and review-friendly
- prefer nouns or short noun phrases like `scheduler`, `lm-head-new`, `npu-template`
- avoid spaces, uppercase letters, and vague names like `misc`, `temp`, `test-branch`
- avoid repeating the type in the topic, such as `feat/feature-x`
- use slash-separated namespace prefixes only when the work clearly belongs to a scoped stream

Scoped branch guidelines:

- use `<namespace>/<type>/<topic>` for team-, model-, or project-scoped work such as `dsv4/feat/rope-dsv4`
- keep the namespace stable and meaningful, not personal or temporary
- use `preview/<topic>` only for preview-track work that intentionally aligns with preview branches upstream
- reserve `release/vX.Y.Z` for release preparation or release-only changes
- avoid direct development on `main` and `release/*` unless the user explicitly asks for it

Examples:

- `feat/skills`
- `feat/lm_head_new`
- `bugfix/scheduler`
- `refactor/npu_template`
- `preview/glm-5`
- `release/v0.9.0`

Notes for xLLM:

- `main` is the default development branch
- `feat/*`, `bugfix/*`, and `refactor/*` appear in current branch usage and are safe defaults
- `preview/*` and `release/*` are long-lived integration branches, not ordinary personal topic branches
- both hyphen and underscore appear in existing history, but prefer hyphens for new branch topics unless matching an established naming family
