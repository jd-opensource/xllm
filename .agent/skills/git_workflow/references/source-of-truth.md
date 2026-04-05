# Source Of Truth

Use these repo files first when the user asks for xLLM-specific Git guidance:

- `README.md`
- `CONTRIBUTING.md`
- `RELEASE.md`
- `.github/workflows/check_format.yml`
- `.pre-commit-config.yaml`
- `.github/CODEOWNERS`

Guidance priority:

1. direct user request
2. current repo files and branch reality
3. visible remote repo conventions
4. older internal notes or remembered habits

When repo docs, local repo state, and user wording disagree:

- say the conflict explicitly
- prefer the most concrete source available
- avoid inventing rules that are not visible in the public repo

Common examples of rules you should not assume without evidence:

- rebase-only or squash-only merge requirements
- mandatory reviewer counts beyond what `CODEOWNERS` implies
- old naming patterns such as `features/*`, `release_0.1.0`, or `v0.1.0-rc0`
