import argparse
import sys
from pathlib import Path

from virda.config import (
    VirdaSettings,
    get_virda_settings,
    resolve_ese_config,
    resolve_stage2_config,
)
from virda.ese.pca_ese_builder import PCAESEBuilder
from virda.models.ese_mesh import ESEMesh
from virda.models.stage1_result import Stage1Result
from virda.pipelines.helpers import setup_pipeline_logging
from virda.pipelines.stage1 import Stage1PipelineBuilder
from virda.pipelines.stage2 import Stage2PipelineBuilder


def run(
    nifti_path: str | Path | None = None,
    project_dir: str | Path | None = None,
    fiducials_path: str | Path | None = None,
) -> tuple[Stage1Result, ESEMesh | None]:
    """Run the full VIRDA pipeline: Stage 1 → 2 → 3.

    Parameters
    ----------
    nifti_path
        Path to T1-weighted NIfTI
    project_dir
        Path to output directory
    fiducials_path
        Path to manual fiducials file

    Returns
    -------
    tuple[Stage1Result, ESEMesh | None]
        Stage 1 result and the ESE surface when ESE is configured, else None.
    """
    settings = get_virda_settings()
    stage1_result = (
        Stage1PipelineBuilder.from_settings(
            settings=settings,
            nifti_path=nifti_path,
            project_dir=project_dir,
            fiducials_path=fiducials_path,
        )
        .build()
        .run()
        .get_store_notnull(Stage1Result)
    )
    ese_mesh = _run_stage2(settings, stage1_result, project_dir)
    return stage1_result, ese_mesh


def _run_stage2(
    settings: VirdaSettings,
    stage1_result: Stage1Result,
    project_dir: str | Path | None,
) -> ESEMesh | None:
    """Build and export the ESE surface when ESE is configured, otherwise None."""
    ese_config = resolve_ese_config(settings)
    if ese_config is None:
        return None

    stage2_config = resolve_stage2_config(settings)
    if stage2_config is None:
        return None

    resolved_project_dir = project_dir or settings.project_dir
    if resolved_project_dir is None:
        raise ValueError(
            "Project directory path not provided. "
            "Pass it as an argument or set the PROJECT_DIR environment variable."
        )
    project = Path(resolved_project_dir)

    stage2_pipeline = Stage2PipelineBuilder(
        ese_builder=PCAESEBuilder(
            config=stage2_config,
            ese_offset_mm=ese_config.ese_offset_mm,
        ),
        stage2_config=stage2_config,
        scalp_mesh=stage1_result.mesh,
        project_dir=project,
        logger=setup_pipeline_logging(project, "stage_2"),
    ).build()
    context = stage2_pipeline.run()
    return context.get_store_notnull(ESEMesh)


def _parse_cli_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse kebab-case path flags; leave everything else to pydantic-settings."""
    parser = argparse.ArgumentParser(
        prog="virda",
        description="Run the VIRDA electrode localization pipeline (Stage 1).",
    )
    parser.add_argument("--nifti-path", dest="nifti_path", help="Path to the T1 NIfTI scan.")
    parser.add_argument("--project-dir", dest="project_dir", help="Path to the output directory.")
    parser.add_argument(
        "--fiducials-path", dest="fiducials_path", help="Path to the manual fiducials file."
    )
    return parser.parse_known_args()


def main() -> None:
    args, remaining = _parse_cli_args()
    sys.argv = [sys.argv[0], *remaining]
    stage1_result, ese_mesh = run(
        nifti_path=args.nifti_path,
        project_dir=args.project_dir,
        fiducials_path=args.fiducials_path,
    )
    print(f"Stage 1: mesh with {len(stage1_result.mesh.vertices)} vertices")
    if ese_mesh is not None:
        print(f"Stage 2: ESE mesh with {len(ese_mesh.vertices)} vertices")
