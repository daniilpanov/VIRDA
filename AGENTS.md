# AGENTS.md

## Toolchain

- Python >=3.14, `uv` workspace: core `virda` (root manifest, `src/virda`) + `virda-gui` (`src/virda_gui`). `uv sync --all-extras --all-packages`; run everything via `uv run <cmd>` (CI appends `--frozen`).
- Full gate before committing: `uv run pre-commit run --all-files` (`black`, `isort`, `ruff check --fix`, `ruff format`, `mypy src/virda/ tests/` — strict, line-length 100).
- Tests: `uv run pytest tests/` (addopts inject `--cov=src/virda --cov-report=term-missing`). Single file: `uv run pytest tests/test_<name>.py -v`. `integration`-marked tests are skipped unless `--run-integration`.

## Architecture

- `src/virda` is the pure core; public API is `virda.ops` (option dataclasses, `ScalpSurface` bundle, pure atoms: `generate_scalp_surface`, `clean`, `smooth`, `decimate`, `generate_ese`, `localize`). File parsing/serialization lives in `io/importers` / `io/exporters`; domain types in `models/`, algorithms in `mesh/`, `ese/`, `localization/`, `segmentation/`, `geometry/`, `qc/`. There is no `pipelines/` package and no core CLI.
- `src/virda_gui` is the PySide6 app (pyvista for 3D) with console scripts `virda-gui`, `virda-gui-viewer`, `virda-gui-html` (defined in `src/virda_gui/pyproject.toml`).
- On-disk patient-project layout is specified in `PATIENT_PROJECT_FORMAT.md` — read it before touching pipeline I/O or the GUI run/export flow.

## Conventions (see CONTRIBUTING.md)

- English (ASCII) only: code, comments, docs, commits, branches.
- Commits: `<prefix>(<module>): <subject>` with `prefix` in `feat|fix|hotfix|chore|refactor`; 1 commit = 1 microfeature (code + its tests together); every commit must leave tests + pre-commit green. No pushing to `master` directly — branches `feature|improve|fix/[<scope>/]<name>` + PR.

## Gotchas

- Stale CI references, don't follow them: `ci.yml`'s integration job runs non-existent `tests/test_stage3_e2e.py`; `test-data.yml` invokes a removed `virda` CLI (`uv run --no-sync virda --nifti-path ...` fails — root manifest defines no scripts and there is no `__main__.py`).
- GUI tests: most avoid `QApplication` entirely; widget tests (`test_gui_app.py`, `test_live_edit_widgets.py`) set `QT_QPA_PLATFORM=offscreen` themselves, but the machine needs Qt system libs to import PySide6 (`libegl1 libgl1 libxkbcommon0` on Ubuntu; see `.github/actions/install-qt-libs`).
- `TASK.md` and `.scratch/` are git-ignored local notes; `wt/`, `dist/`, `build/`, `htmlcov/`, `.coverage` are generated/scratch — ignore them.
