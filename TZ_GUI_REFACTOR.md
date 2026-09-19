# ТЗ: очистка GUI от пайплайнов, конфигов и логов (ветка develop)

> **ВАЖНО ДЛЯ COMPACTION / СЖАТИЯ КОНТЕКСТА:**
> Перед продолжением работы ОБЯЗАТЕЛЬНО прочитай этот файл целиком
> (`TZ_GUI_REFACTOR.md` в корне репозитория). Он фиксирует все решения
> пользователя и текущее состояние работы. Не продолжай, не прочитав его.

## Цель
Достроить реархитектуру VIRDA на ветке `develop`: полностью убрать из GUI
принцип пайплайнов, `pipeline_config.json`, конфигурационные файлы, логи,
авто-запуск и HTML-экспорт. Всё управляется ТОЛЬКО GUI. Добавить новый
импорт файлов с авто-детекцией роли и новый способ генерации scalp-меша.

## Текущее состояние (зафиксировано)
- Ветка `develop`, дерево чистое. Последний коммит: `b35ab6f`.
- Коммиты реархитектуры (локально): `2f5ed1f` (ops package),
  `2c61061` (cut pipeline/config layers), `a61d517` (cut config models, npy,
  auto fiducial), `b35ab6f` (порты тестов на чистый ops/io API).
- PUSH: пользователь уже запушил на другом устройстве ранее; в самом конце
  работы я делаю ещё один `git push origin develop` (если не пройдёт — просто
  сообщить, не паниковать).
- Валидация СТРОГО статическая: ничего не запускать (никакие тесты/приложения)
  — только `python -m ast`, `git diff --check`, import-свип, `rg`.
- Текущие кодовые факты (проверено чтением):
  - `pipeline_runner.py` удаляется; `logging.py` удаляется; `tabs/config_tab.py`
    удаляется.
  - `mesh_processing_tab.py`: есть smooth/decimate/ESE-превью и «Save to
    project» (пишет PLY + `pipeline_config.json` через `_write_mesh_parameters`
    — её удалить). Нужно добавить генерацию scalp-меша из NIfTI.
  - `editors_tab.py`: `FiducialsEditor` имеет combo `Coordinate system`
    (FRAME_IDS: scanner_ras/voxel/cras), `input_frame()`, `set_rows()`.
    Чистые хелперы фреймов в `viewer/frames.py`
    (`world_to_frame_matrix`, `frame_to_world_matrix`, `transform_points`).
  - `importing.py`: `ROLE_REGISTRY`, `ImportRole`, `import_target`, `import_file`.
  - `main_window.py`: IdwWindow с ConfigTab, PipelineRunner, логом, HTML-экспортом — всё это убрать.
  - `sidebar.py`: кнопка Run pipeline, сигнал `runPipelineRequested`,
    подменю «Import artifact...».
  - `constants.py`: константы `DEFAULT_PIPELINE_CONFIG_FILENAME`,
    `CONFIG_KEY_TO_INPUT`, `CONFIG_KEY_TO_ADVANCED` — убрать; `ADVANCED_FIELD_DEFAULTS`
    — почистить до используемых ключей.
  - `state.py`: убрать `log_queue`, `stage3_summary`, `coordsystem`.
  - `tests/test_gui_app.py`: переписать (см. раздел «Тесты»).
  - HTML-экспорт: модуль `virda_gui/export/html_export.py` + CLI `virda-gui-html`
    в `pyproject.toml` + использование в `.github/workflows/test-data.yml` —
    МОДУЛЬ ОСТАВИТЬ, убрать только GUI-кнопку/сигнал/потоки в main_window.

## Решения пользователя (обязательны, не переспрашивать)
1. **NPY-меш НЕ делаем.** Один NPY-файл (только `faces`, или только `vertices`,
   или `adjacency`) не содержит полной геометрии — меш по одному файлу не
   восстановить. Такие файлы ОТБРАСЫВАЮТСЯ с ошибкой «cannot build a mesh».
   Импорт меша — ТОЛЬКО PLY. Из PLY автоматически ничего не строить; все
   экспортные NPY пишет только пользователь вручную. NPY-роли `scalp_vertices`,
   `scalp_faces`, `scalp_face_adjacency` из реестра импорта УДАЛИТЬ.
   Остаётся `normals` (npy-файл точек, `ese/normals.npy`).
2. **Никаких `pipeline_config.json` / `input/config.json` НИГДЕ.** Удалить:
   роль `config` из реестра, селектор «Config file», кнопки «Save Config»,
   `write_pipeline_config`, `serialize_config_for_save`, `import_pipeline_config`
   (использование), `_write_mesh_parameters`. Всё контролируется ТОЛЬКО GUI.
3. **Нельзя ничего собирать автоматически** — нет авто-запуска при открытии
   проекта или импорте. Никаких триггеров «запустить цепочку».
4. **Никаких логов и пайплайнов.** Убрать `LogViewer`, `log_viewer`,
   `log_queue`, `PipelineRunner`, кнопки «Run Pipeline» (sidebar + form),
   вкладку «Run Pipeline», обработчики `services/logging.py`. Сообщения —
   только статус-бар/диалоги (`QMessageBox`), либо без них.
5. **ConfigTab удалить целиком и рассредоточить:**
   - плотность/сглаживание/ESE offset уже в MeshProcessingTab (оставить);
   - `ElectrodeGroupRow`/оверлеи электродов → компактная панель в 3D-viewer
     вкладку (env: `state.electrode_rows`, `palette_index`, `ELECTRODE_PALETTE`);
   - «Force cRAS conversion» checkbox (был `electrodes_cras_check`) — перенести
     в эту же панель, состояние в `AppState` (например `state.electrodes_cras`);
   - Advanced Settings (`AdvancedSettingsDialog`) → кнопка в EditorsTab
     («Localization settings...»); диалог почистить от секций этапов
     (Segmentation otsu/closing, mesh smoother/density/voxel); оставить
     только используемые GUI клавиши: seal_enabled/seal_radius,
     cleaner_min_vertices/cleaner_merge_digits, neighborhood ESE,
     residual_threshold_mm, calibrate_ese_offset; `ADVANCED_COMBO_FIELDS`
     удалить если не нужны комбо.
   - `mesh_voxel_size_mm` убрать полностью (никто не читает).
   - `_collect_viewer_kwargs`: nifti — глоб `input/*.nii*` из проекта;
     проект — `state.last_project_dir`; «Add stage3 electrodes group» больше
     не нужен (файл не создаётся без пайплайна) — систему групп оставить ручной.
6. **HTML-экспорт из GUI убрать** (кнопка, сигнал `exportHtml`, потоки в runner).
7. **Генерация scalp-меша через GUI** — кнопка в MeshProcessingTab
   «Generate scalp mesh from NIfTI...»: выбор NIfTI (по умолчанию
   `project/input/*.nii*`), `import_nifti` → `generate_scalp_surface(mri,
   sealing)` → `clean(surface.mesh, cleaning)`, где sealing/cleaning строятся
   из `state.advanced` (seal_enabled/seal_radius/cleaner_min_vertices/
   cleaner_merge_digits, с дефолтами). Результат задаёт `_base_mesh`;
   smooth/decimate продолжат работать живьём. Ошибка → `QMessageBox.critical`.
8. **Import files** (требование пользователя №1):
   - В контекстном меню sidebar пункт «Import files...» (сигнал
     `importFilesRequested`), вместо подменю «Import artifact...».
     Доступен при создании/открытии проекта (движок контекстного меню уже есть).
   - `detect_role(path) -> ImportRole | None` (Qt-free) по имени/расширению/
     содержимому: `.nii/.nii.gz` → nifti; `.ply` (имя содержит «ese» →
     ese_mesh, иначе None/неоднозначно); `*normals*.npy` → normals;
     `.json` по структуре (топ-уровень list с `ese_coords|coords` →
     electrodes; dict с `"electrodes"` → measurements; `"fiducials"` →
     fiducials); `.npy` фигу — None (для меша невалиден, ошибка).
   - Фолбэк: `QInputDialog.getItem` со списком из 6 ролей:
     nifti / mesh / ese mesh / normals / localized electrodes / measurements.
   - `validate_import_source(role, path)` (Qt-free): mesh/ese_mesh →
     `import_scalp_mesh` (trimesh); nifti → `nib.load` БЕЗ `get_fdata`;
     normals → `np.load` + shape (N,3); measurements → `import_measurements`;
     electrodes → проверка JSON-схемы; ошибка — `ValueError` с понятным текстом.
   - Неверный файл → `QMessageBox.critical("Import error", ...)`.
9. **Fiducials: пересчёт при смене системы координат:**
   - `FiducialsEditor` emits `inputFrameChanged(old, new)`.
   - В `main_window`: если конверсия невозможна (voxel без affine; cras без
     offset; нет viewer) → warning + вернуть combo на старое значение.
   - Иначе `M = world_to_frame_matrix(new, affine, cras) @
     frame_to_world_matrix(old, affine, cras)`; применить к координатам всех
     строк, обновить ячейки; per-row `coordinate_system`: scanner_ras→world,
     voxel→voxel, cras→world.
   - Добавить Qt-free хелпер в `viewer/frames.py`:
     `frame_to_frame_matrix(old, new, affine, cras)` + покрыть тестом.

## План изменения файлов
### Удалить
- `src/virda_gui/services/pipeline_runner.py`
- `src/virda_gui/services/logging.py`
- `src/virda_gui/tabs/config_tab.py`

### Изменить
- `src/virda_gui/constants.py`: убрать `DEFAULT_PIPELINE_CONFIG_FILENAME`,
  `CONFIG_KEY_TO_INPUT`, `CONFIG_KEY_TO_ADVANCED`, `ADVANCED_COMBO_FIELDS`
  (если не используется), `mesh_voxel_size_mm`, otsu/closing-ключи из
  `ADVANCED_FIELD_DEFAULTS`. Оставить: `ELECTRODE_PALETTE`,
  `DEFAULT_FIDUCIALS_FILENAME`, `DEFAULT_MEASUREMENTS_FILENAME`,
  `PROJECT_ARTIFACT_DIRS`, укороченный `ADVANCED_FIELD_DEFAULTS`.
- `src/virda_gui/importing.py`: новый `ROLE_REGISTRY`
  (mesh/ese_mesh/normals/electrodes/nifti/measurements/fiducials);
  `detect_role(path)`; `validate_import_source(role, path)`;
  `import_target`/`import_file` без изменений.
- `src/virda_gui/sidebar.py`: сигнал `importFilesRequested`; убрать
  `runPipelineRequested`, кнопку «Run pipeline»; контекстное меню:
  пункт «Import files...».
- `src/virda_gui/state.py`: убрать `log_queue`, `stage3_summary`, `coordsystem`;
  добавить `electrodes_cras: bool = False`.
- `src/virda_gui/main_window.py`: большая переработка — без pipeline/лог/конфиг/
  html; Import files flow; fiducials frame recalc; панель electrode overlays
  в viewer вкладке; статус-бар сообщения; `open_project` не открывает никакую
  вкладку принудительно.
- `src/virda_gui/tabs/mesh_processing_tab.py`: кнопка генерации из NIfTI;
  удалить `_write_mesh_parameters` и запись конфига; статус → сигнал, который
  main_window пробрасывает в statusBar.
- `src/virda_gui/tabs/editors_tab.py`: `inputFrameChanged` сигнал + API
  `set_input_frame` (есть) + кнопка «Localization settings...» (сигнал
  `advancedRequested`); каждый row `coordinate_system` обновляется при recalc.
- `src/virda_gui/dialogs/advanced_settings.py`: почистить секции, оставить
  seal/cleaner/neighborhood/localization; убрать комбо-поля (если убрали
  `ADVANCED_COMBO_FIELDS`) и `step_size_for_voxel_size` чтение.
- `src/virda_gui/viewer/frames.py`: добавить `frame_to_frame_matrix`.
- `src/virda_gui/widgets.py`: `LogViewer` можно оставить в модуле (не удалять,
  чтобы не ломать импорты preview?), НО убрать использование. Решить в ходе
  правок, минимально безопасно.

### Тесты (`tests/test_gui_app.py`)
- Убрать: импорты `import_pipeline_config`, `PipelineRequest`,
  `ConfigTab/collect_config/density_percent_from_state/serialize_config_for_save/
  write_pipeline_config`, тесты про config-файл/вкладку, эмиссию
  `runPipelineRequested`, `_config_tab.*` assertions, `_default_config_save_path`.
- Заменить offscreen-тесты открытия проекта: тот «Run Pipeline» удалён, теперь
  `open_project` не добавляет вкладок.
- Добавить тесты: `detect_role` (все архетипы файлов), `validate_import_source`
  (валидный PLY, кривой PLY, нормали NPY, неверная форма NPY, measurements,
  electrodes, nifti), `frame_to_frame_matrix` (identity, world↔voxel на
  известном affine), mesh-генерация на уровне атомов (без Qt, создать
  мини-NIfTI через nibabel в tmp_path, прогнать `generate_scalp_surface`+
  `clean` и проверить ScalpMesh).
- Проверить, что ничего не осталось от «pipeline», «config file», «log».
- `test_ide_window_constructs_and_manages_project_offscreen`: number of tabs
  после `open_project` == результату «никакая вкладка не открыта» (т.е. 0 +
  добавленная пользовательская). Внимательно с `_close_tab(1)` etc.

## Проверка (только статическая)
1. `git add -A && git diff --cached --check` — нет пробельных ошибок.
2. Свип: структура импортов без unresolved (перечисленные модули импортируются).
3. `python -m compileall` НЕ запускать без необходимости? Можно, это не
   исполнение тестов — допустимо. НО ничего не запускать если не уверены.
   Безопасно: ast-парсер script (как раньше) — использовать обязательно.
4. Убедиться, что `rg -i "pipeline|log_viewer|log_queue|config_file|runPipeline|Run Pipeline|exportHtml|stage3_summary|pipeline_config" src tests` не даёт ложных следов кроме разрешённых (html_export.py, viewer docstrings, константа в pyproject/workflow).

## Порядок работы
1. Написать файл `TZ_GUI_REFACTOR.md` (это делаем).
2. constants → importing → sidebar → state → frames → main_window →
   mesh_processing_tab → editors_tab → advanced_settings → удалить три файла.
3. tests/test_gui_app.py переписать.
4. Статическая проверка (см. выше).
5. `git add` + commit (conventional, стиль репозитория),
6. `git push origin develop` — не пройдёт — просто сообщить.