# PR And Review

## Pull Request Guidance

The public repo guidance is lightweight:

- `README.md` and `CONTRIBUTING.md` ask contributors to fork, create a branch, and send a pull request
- keep PRs focused and easy to review, even though the public docs do not publish a hard line-count limit
- write commit messages and PR descriptions in clear English
- avoid unnecessary merge noise in branch history; prefer a clean linear history when practical

## Target Branch

- use `main` unless this is explicitly a release or backport task

## Review Expectations

Use `.github/CODEOWNERS` as the visible review signal:

- changes under `/xllm/` have listed code owners
- expect owner review or owner attention for those paths
- if the user asks who should review a change under `/xllm/`, check `CODEOWNERS`

## Quick Checklist

- PR target is `main` unless this is a backport or release task
- PR is focused and clearly described
- review expectations are checked through `CODEOWNERS`
