# Agent Instructions for Codex

## Operating Principles
1. **Always plan first.** Provide a brief plan before any changes and do the hardest thinking during the planning phase.
2. **Highlight critical points.** When proposing tasks, call out steps that could introduce side effects (e.g., migrations, API changes, backward-compatibility concerns).
3. **Ask one question at a time.** If a decision is needed from a user, ask a single, clear question before proceeding.

## Branch Naming
- **Format:** `<type>/<short-description>`
- **Examples:**
  - `fix/one-off-bug-gh134`
  - `enh/add-bids-validator`
  - `doc/update-readme`
- If working from a tracked issue, include the issue number in the branch name (e.g., `fix/segfault-gh134`).

## Commit Messages (Conventional Commits)
- **Must strictly follow** Conventional Commits: https://www.conventionalcommits.org/en/v1.0.0/
- Examples:
  - `fix(parser): handle empty token stream`
  - `docs(readme): add quickstart example`
  - `chore(ci): pin action versions`

## Pull Request Titles & Descriptions
- **PR Title format:** `TYPE: Short summary`
  - Type is **all caps**, from a conventional-commit-like set (e.g., FIX, ENH, MNT, DOC, CI, CHORE).
  - **No scope parenthetical** in PR titles.
- **Examples:**
  - `FIX: Address one-off bug in traversing list`
  - `DOC: Complete docstring with missing arguments`
- **PR Description should include:**
  - Summary of changes
  - Motivation / context
  - Testing performed (exact commands)

## Linting & Pre-Change Validation
- **Always run linting before proposing changes.**
- This repository uses the GitHub Actions linting command:
  - `pipx run ruff format --diff`
- If linting cannot run due to environment constraints, state the limitation and suggest a retry in a fully configured environment.
