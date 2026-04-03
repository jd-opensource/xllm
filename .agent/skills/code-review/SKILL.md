---
name: code-review
description: Review code changes for quality, security, performance, and correctness following project-specific standards. Use when reviewing pull requests, examining git diffs, or when the user asks for a code review. This skill should be used proactively — when the user asks for a review without specifying commits, automatically detect the current branch and diff against the main branch.
---

# Code Review

## Workflow

### Step 1: Determine the diff

If the user provides explicit SHAs or a PR link, use those. Otherwise, **auto-detect**:

```bash
# Fetch latest remote state
git fetch origin main --quiet

# Detect current branch
CURRENT_BRANCH=$(git branch --show-current)

# Find the merge base with origin/main
MERGE_BASE=$(git merge-base origin/main HEAD)

# Show what changed
git diff --stat $MERGE_BASE..HEAD
git diff $MERGE_BASE..HEAD
```

If `CURRENT_BRANCH` is `main`, warn the user and ask which commits to review.

### Step 2: Read project standards

Read [custom-code-style.md](references/custom-code-style.md) for project-specific coding style.

### Step 3: Review against the checklist

**Correctness:**
- Logic handles edge cases and boundary conditions
- Error handling is comprehensive (no silent failures)
- Type safety maintained (no unsafe casts, proper use of `std::optional`)
- Resource lifecycle correct (RAII, no leaks, proper cleanup order)

**Architecture:**
- Clean separation of concerns, no layer violations
- Dependencies flow in the correct direction
- Changes align with existing patterns in the codebase
- No unnecessary coupling introduced

**Performance & Concurrency:**
- No performance regressions on hot paths
- Thread safety: proper locking, no data races
- CUDA/NPU kernels: memory coalescing, occupancy, sync correctness
- No unnecessary copies of large objects (tensors, vectors)

**Testing:**
- Tests verify actual logic, not just mock wiring
- Edge cases and error paths covered
- Integration tests for cross-component changes

**Production Readiness:**
- Backward compatibility maintained (or breaking changes documented)
- Migration strategy for schema/config changes
- No hardcoded values that should be configurable

### Step 4: Output findings

Use the format below.

## Output Format

### Strengths
[Specific things done well, with file:line references]

### Issues

#### Critical (Must Fix)
[Bugs, security holes, data loss risks, broken functionality]

#### Important (Should Fix)
[Architecture problems, missing error handling, test gaps, performance issues]

#### Minor (Nice to Have)
[Style, optimization opportunities, documentation improvements]

**Each issue must include:**
- **File:line** reference
- **What** is wrong
- **Why** it matters
- **How** to fix (if not obvious)

### Recommendations
[Broader improvements for code quality, architecture, or process]

### Assessment

**Ready to merge?** [Yes / No / With fixes]

**Reasoning:** [1-2 sentence technical assessment]

## Rules

**DO:**
- Apply project-specific style from [custom-code-style.md](references/custom-code-style.md)
- Follow DDD (Domain Driven Design) principles, and keep the codebase clean and maintainable
- Categorize by actual severity (not everything is Critical)
- Be specific with file:line references
- Explain WHY issues matter
- Acknowledge strengths
- Give a clear verdict

**DON'T:**
- Approve without thorough review
- Mark nitpicks as Critical
- Give feedback on code not in the diff
- Be vague (e.g., "improve error handling" without specifics)
