# Branch And PR

## Current Repo Conventions

The current public xLLM repo shows these conventions:

- the default development branch is `main`
- contributors are asked to fork the repository, create a branch, and send a pull request
- active long-lived branches include `preview/*` and `release/vX.Y.Z`
- topic branches commonly use shapes such as `feat/<topic>` and `bugfix/<topic>`

## Daily Development Flow

Follow this sequence unless the user asks for a different workflow:

1. fork the upstream repository
2. sync local `main` with `upstream/main`
3. create a focused topic branch from `main`
4. implement the change
5. run formatting and the narrowest relevant validation
6. commit in clear English
7. push to the fork
8. open a PR to upstream `main`

Recommended branch naming:

- prefer a short descriptive topic branch
- match existing repo style when possible, such as `feat/<topic>` or `bugfix/<topic>`
- use `preview/<topic>` only when intentionally aligning with preview work
- avoid direct development on `main` and `release/*`

Example commands:

```bash
git fetch upstream
git checkout main
git pull --rebase upstream main
git checkout -b feat/<topic>

# after development
git add <files>
git commit -m "feat: add <change summary>."
git push origin feat/<topic>
```

## Pull Request Guidance

The public repo guidance is lightweight:

- `README.md` and `CONTRIBUTING.md` ask contributors to fork, create a branch, and send a pull request
- keep PRs focused and easy to review, even though the public docs do not publish a hard line-count limit
- write commit messages and PR descriptions in clear English
- avoid unnecessary merge noise in branch history; prefer a clean linear history when practical

Default target branch:

- use `main` unless this is explicitly a release or backport task

## Review Expectations

Use `.github/CODEOWNERS` as the visible review signal:

- changes under `/xllm/` have listed code owners
- expect owner review or owner attention for those paths
- if the user asks who should review a change under `/xllm/`, check `CODEOWNERS`

## Quick Checklist

- target branch is `main` unless this is a backport or release task
- topic branch name is descriptive and matches current repo style
- PR is focused and clearly described
- owner review expectations are checked through `CODEOWNERS`
