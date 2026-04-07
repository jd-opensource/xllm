# Backport Flow

When a released line needs a bugfix, prefer this flow:

1. land the fix on `main` first unless the user explicitly needs a direct hotfix flow
2. cherry-pick or backport the fix to the matching `release/vX.Y.Z` branch
3. update release content as needed on that release branch
4. create the next patch tag for that release line

Example shape:

```bash
# land on main first
git checkout main
git pull --rebase upstream main
git checkout -b bugfix/<topic>
git commit -m "bugfix: fix <summary>."

# then backport
git checkout release/v0.9.0
git pull --rebase upstream release/v0.9.0
git cherry-pick <bugfix_commit>
git tag v0.9.1
```

## Quick Checklist

- backport from a commit already landed on `main` when possible
- cherry-pick onto the matching `release/vX.Y.Z` branch
- use the next semantic patch tag for the release line
