# CONTRIBUTING

## Language

Write everything in English only: code, comments, documentation, commit messages,
branch names, and any other text that ends up in the repository. Prefer ASCII
characters.

## Branches

- `feature/<scope>/<name>` or `feature/<name>` — new functionality;
- `improve/<scope>/<name>` or `improve/<name>` — improvement or refactoring;
- `fix/<scope>/<name>` or `fix/<name>` — bugfix.

`<scope>` is optional, include it when it helps to tell branches apart. Examples
from the history: `feature/stage1/fiducials`, `improve/pipelines-architecture`,
`fix/mesh/add-seal`.

Do not push directly to `master`. All changes go through a dedicated branch and a pull request.

## Commits

- 1 commit = 1 microfeature. Do not mix unrelated code (e.g. a model and the code
  that uses it belong to separate commits); avoid large commits — prefer many small ones.
- Every commit must leave the code in a working state: all tests green and pre-commit passing.
- Format: `<prefix>(<module>): <subject>`, where:
  - `prefix` is one of: `feat` / `fix` / `hotfix` / `chore` / `refactor`;
  - `<module>` is optional and goes in parentheses;
  - `<subject>` explains what, why, and the reasoning.
- Skip the body unless the change is genuinely complex — then add a short
  explanation. Do not list changed files (the diff already shows them). Use line
  breaks only where they carry meaning; style is a single line or a list.

Examples:

    feat(export): copy source NIfTI into 'input' directory in 'patient_project'
    refactor: use the new 'PipelineController'
    fix(mesh): 'trimesh_mesh.split' returns only watertight by default - use 'only_watertight=False'

## Tests

Create tests in a single commit with a microfeature. Avoid tautological and silly tests.
Don't test the library: if you're using pydantic, DO NOT write `assert YourModel(field=0).field == 0`.
