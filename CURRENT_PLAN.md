# Current Plan: Closing Stage 1 gaps

Stage 1 реализована почти полностью, но есть заметные gap'ы относительно
`docs/task/VIRDA_Stage1_Head_Surface_Mesh_Generation.md`. План закрывает их.

## Коммит 0 — `.gitignore` (первым, отдельный коммит)
- Добавить `research/` в `.gitignore`.
- `docs/plan/` в коммит НЕ включаем.
- После финальных коммитов — `git push` ветки `feature/stage1/fiducials-ese-export`.

## 1. Реструктуризация `io/qc/` → по модулям
- **`src/virda/geometry/`** (новый): перенос `io/qc/geometry.py` → `transforms.py`
  (`fiducials_world_coordinates`, `mesh_voxel_coordinates`).
- **`src/virda/visualization/`** (новый): перенос `slices.py`, `render.py`, `viewer.py`;
  `__init__.py` с оркестратором `write_visual_artifacts(result, output_dir, mesh_path, with_html)`
  (визуальная часть старого `run_qc`).
- **`src/virda/qc/`** (новый, в корне): `checks.py` + `__init__.py` с `run_checks(result) -> report`.
- Удалить `src/virda/io/qc/`. Обновить импорты в `stage1_exporter.py`, `tests/test_qc.py`,
  `io/__init__.py` (если реэкспортирует).

## 2. ESE — настройка через `VirdaSettings` (§10.3)
- `config.py`: поля `n_electrodes`, `ese_offset_mm`, `ese_reference`.
- `main.py`: `ESEConfig(...)` из `settings`.
- `_pipeline_config` в `stage1_exporter.py`: `"ese"` из `ESEConfig`, плоские `ese_*` исключить из дампа.

## 3. `ScalpMesh`: adjacency + metadata (§7.3, §20)
- `scalp_mesh.py`: необязательные `face_adjacency: np.ndarray | None = None`,
  `coordinate_system: str = "world"`, `metadata: dict = field(default_factory=dict)`.
- `mesh_cleaner.py`: заполнять `face_adjacency` из `trimesh_mesh.face_adjacency`.
- Экспорт `mesh/scalp_face_adjacency.npy` + `n_adjacency_edges` в манифесте.

## 4. Автоматические QC-проверки + логирование (§13.1, §16)
- `qc/checks.py`: `check_mri` (affine/spacing/orientation), `check_mesh` (непустой,
  валидные индексы, не слишком разрежен), `check_components(mask)` (несколько крупных
  компонентов), `check_fiducials` (перенос из `stage1_exporter.fiducial_qc`),
  `run_checks(result)` → отчёт.
- `stage1_exporter.py`: `quality_control/report.json`; `logging` → `logs/stage1.log`
  (вместо `print`).

## 5. Структура `patient_project/` (§15) + NIfTI-маска
```
patient_project/
  input_mri/provenance.json
  segmentation/head_mask.nii.gz     # export_segmentation() в io/exporter/nifti_exporter.py
  mesh/scalp.ply + scalp_face_adjacency.npy
  fiducials/fiducials.json
  config/pipeline_config.json
  quality_control/report.json + qc_overlay_*.png + qc_3d_*.png + head_viewer.html
  logs/stage1.log
  stage1_result.json                # манифест (§14) в корне
```

## 6. Тесты и верификация
- NIfTI-маска: round-trip в `test_stage1_pipeline` (`nib.load` → shape/affine/voxel_count),
  unit-тесты `export_segmentation` (пустая маска, affine), QC-проверка файла vs in-memory.
- `tests/test_qc_checks.py`, `tests/test_scalp_mesh.py`, тест ESE через settings;
  обновить `test_qc.py` и пути в `test_stage1_pipeline.py`.
- `uv run pytest`, `ruff check src tests`, `mypy src`; санity-прогон на icbm152.

## Вне скоупа
- DICOM-загрузчик (в спеке «preferably»).
- Интерактивное редактирование фидуциалов.
