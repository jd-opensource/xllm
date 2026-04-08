# Development Flow

Follow this sequence unless the user asks for a different workflow:

1. fork the upstream repository
2. sync local `main` with `upstream/main`
3. create a focused topic branch from `main`
4. implement the change
5. run formatting and the narrowest relevant validation
6. commit in clear English
7. push to the fork
8. open a PR to upstream `main`

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

## Quick Checklist

- branch from `main` unless this is a release or backport task
- keep the branch focused on one change
- run the narrowest relevant validation before commit
- push to your fork before opening the PR
