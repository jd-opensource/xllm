# Release Layout

Observed public release layout:

- release branches use `release/vX.Y.Z`
- observed release branches include `release/v0.6.0`, `release/v0.7.0`, `release/v0.8.0`, and `release/v0.9.0`
- release notes are tracked in `RELEASE.md`
- public tags use semantic version tags such as `v0.9.0`
- patch tags use normal patch versions such as `v0.7.1` and `v0.7.2`, not `-rcN`

## Naming Guardrails

- do not switch to `release_0.1.0` branches unless the user explicitly wants an older internal workflow
- do not assume `v0.1.0-rc0` style tags unless the user explicitly asks for them
- prefer the next semantic patch tag for bugfix releases
