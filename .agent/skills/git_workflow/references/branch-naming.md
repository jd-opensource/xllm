# Branch Naming

## Current Repo Conventions

The current public xLLM repo shows these conventions:

- the default development branch is `main`
- active long-lived branches include `preview/*` and `release/vX.Y.Z`
- topic branches commonly use shapes such as `feat/<topic>` and `bugfix/<topic>`

## Recommended Naming

- prefer a short descriptive topic branch
- match existing repo style when possible, such as `feat/<topic>` or `bugfix/<topic>`
- use `preview/<topic>` only when intentionally aligning with preview work
- avoid direct development on `main` and `release/*`

## Examples

- `feat/git-workflow-skill`
- `bugfix/npu-kernel-dispatch`
- `preview/speculative-batching`
- `release/v0.9.0`
