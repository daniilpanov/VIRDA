# AGENTS.md

## Toolchain

- Python >=3.14, `uv` workspace: core package `virda` (`src/virda`) + GUI `virda-gui` (`src/virda_gui`). Install everything with `uv sync --all-extras --all-packages`; run any tool via `uv run <cmd>`.
- Lint/format/typecheck are **local pre-commit hooks** (`black`, `isort`, `ruff check --fix`, `ruff format`, `mypy src/virda/ tests/` — strict, line-length 100). Run the full gate before committing:
  `uv run pre-commit run --all-files`
- Tests: `uv run pytest tests/` (coverage of `src/virda` is on by default via `addopts`). Integration-marked tests are skipped unless you pass `--run-integration`. GUI tests avoid `QApplication`; offscreen smoke tests set `QT_QPA_PLATFORM=offscreen` themselves, but the machine needs Qt system libs to install/import PySide6 (`libegl1 libgl1 libxkbcommon0` on Ubuntu; see `.github/actions/install-qt-libs`).

## Architecture (current state)

- `src/virda` is the pure core (`ops/`, `io/importers`, `io/exporters`, `ese/`, `mesh/`, `segmentation/`, `localization/`, `models/`, `qc/`, `fiducials/`, `geometry/`). `src/virda_gui` is the PySide6 app (pyvista for 3D) with console scripts `virda-gui`, `virda-gui-viewer`, `virda-gui-html` (defined in `src/virda_gui/pyproject.toml`).
- **`develop` is mid-refactor** (79 commits ahead of `master`): GUI is being stripped of pipeline/config/log layers in favor of the pure `ops`/`io` API and a project-folder workflow. `TZ_GUI_REFACTOR.md` (Russian) is the authoritative plan — read it fully before touching GUI code. The target architecture and terminology (MRI-world / voxel / head-RAS frames) are in `VIRDA_TECH_SPEC.md`; the on-disk patient-project format is `PATIENT_PROJECT_FORMAT.md`. `src/virda/pipelines/` still exists but is deprecated/bypassing the docs.
- Read repo guidance first: `CONTRIBUTING.md`, `VIRDA_TECH_SPEC.md`, `PATIENT_PROJECT_FORMAT.md`, `TZ_GUI_REFACTOR.md`.

## Conventions (see CONTRIBUTING.md)

- English (ASCII) only: code, comments, docs, commits, branches.
- Commits: `<prefix>(<module>): <subject>` with `prefix` in `feat|fix|hotfix|chore|refactor`; 1 commit = 1 microfeature (code + its tests together); every commit must leave tests + pre-commit green. No pushing to `master` directly — branch names `feature|improve|fix/[<scope>/]<name>` and MR/PR.

## Gotchas

- The core `virda` package has **no CLI** (no `__main__.py`, no script entry in the root `pyproject.toml`). `full-test.sh` and `.github/workflows/test-data.yml` still call a removed `virda` / `python -m virda` CLI — stale, don't follow them. `ci.yml`'s integration job references the removed `tests/test_stage3_e2e.py`, too.
- `research/` is excluded from mypy and contains experimental scripts/data. Real MRI datasets live in `test-data/`. `.coverage`, `htmlcov/`, `dist/`, `build/`, `research/out/` are generated. `PLAN.md`, `TODO.md`, `task1.md` are personal scratch notes (untracked); `feature/` and `wt/` are empty scratch dirs.