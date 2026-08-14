import argparse
import sys
from pathlib import Path

from virda.config import get_virda_settings
from virda.models.stage1_result import Stage1Result
from virda.pipelines.stage1 import Stage1PipelineBuilder


def run(
    nifti_path: str | Path | None = None,
    project_dir: str | Path | None = None,
    fiducials_path: str | Path | None = None,
) -> Stage1Result:
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
    Stage1Result
    """
    return (
        Stage1PipelineBuilder.from_settings(
            settings=get_virda_settings(),
            nifti_path=nifti_path,
            project_dir=project_dir,
            fiducials_path=fiducials_path,
        )
        .build()
        .run()
        .get_store_notnull(Stage1Result)
    )


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
    result = run(
        nifti_path=args.nifti_path,
        project_dir=args.project_dir,
        fiducials_path=args.fiducials_path,
    )
    print(f"Stage 1: mesh with {len(result.mesh.vertices)} vertices")
