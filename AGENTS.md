# Agent instructions

## Working
- Plan before changing code, and flag side effects: API, CLI, outputs, backward compatibility.
- When a decision is needed, ask one question at a time.
- Lint as CI does before proposing changes, or say why you could not:
  `pipx run ruff==0.17.3 check` and `pipx run ruff==0.17.3 format --diff`.

## Branches
`<type>/<short-description>`, with the issue number when there is one:
`fix/segfault-gh134`, `enh/add-bids-validator`, `doc/update-readme`.

## Commits
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/), kept short:
- a one-line subject, e.g. `fix(parser): handle empty token stream`;
- a one-line body saying why.

Only a fundamental change (behaviour, outputs, compatibility) gets a longer body, and it stays terse.
`doc/` branches and `docs:` subjects skip CircleCI's build and test jobs.

## Pull requests
- Title: `TYPE: Short summary`. TYPE in capitals (FIX, ENH, MNT, DOC, CI, CHORE), no scope.
- Description, brief and to the point: what changes and why, in a few bullets, then the exact test commands.
