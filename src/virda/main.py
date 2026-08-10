import sys
from pathlib import Path

from virda.config import get_virda_settings
from virda.io.exporter.json_io import load_fiducials
from virda.io.exporter.stage1_exporter import Stage1Exporter
from virda.io.loader.nifti_loader import NiftiLoader
from virda.mesh.contracts import MeshSmoother
from virda.mesh.laplacian_smoother import LaplacianSmoother
from virda.mesh.mesh_cleaner import TrimeshCleaner
from virda.mesh.taubin_smoother import TaubinSmoother
from virda.models.ese_config import ESEConfig
from virda.models.fiducial import Fiducial
from virda.models.stage1_result import Stage1Result
from virda.pipelines.stage1 import Stage1Pipeline
from virda.segmentation.head_segmenter import OtsuHeadSegmenter


def run(
    nifti_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    fiducials_path: str | Path | None = None,
    skip_fiducials: bool | None = None,
) -> Stage1Result:
    settings = get_virda_settings()

    resolved_path = nifti_path or settings.nifti_path
    if resolved_path is None:
        raise ValueError(
            "NIfTI path not provided. Pass it as an argument or set the "
            "NIFTI_PATH environment variable."
        )
    resolved_output_dir = output_dir or settings.output_dir
    resolved_fiducials_path = fiducials_path or settings.fiducials_path
    resolved_skip_fiducials = skip_fiducials or settings.skip_fiducials

    fiducials = _load_or_require_fiducials(
        resolved_fiducials_path, skip_fiducials=resolved_skip_fiducials
    )

    loader = NiftiLoader()
    segmenter = OtsuHeadSegmenter()

    cleaner = TrimeshCleaner(
        min_component_vertices=settings.cleaner_min_vertices,
        merge_digits=settings.cleaner_merge_digits,
        remove_internal_faces=settings.remove_internal_faces,
        internal_face_method=settings.internal_face_method,
        internal_face_wide_mm=settings.internal_face_wide_mm,
        internal_face_seed_mm=settings.internal_face_seed_mm,
        internal_face_flood_mm=settings.internal_face_flood_mm,
        internal_face_seed_depth_mm=settings.internal_face_seed_depth_mm,
        internal_face_flood_depth_mm=settings.internal_face_flood_depth_mm,
        internal_face_ray_length_mm=settings.internal_face_ray_length_mm,
        fill_small_holes=settings.fill_small_holes,
        fill_small_holes_max_mm=settings.fill_small_holes_max_mm,
        subdivide_max_edge=settings.subdivide_max_edge,
    )

    smoother: MeshSmoother
    if settings.smoother_type == "taubin":
        smoother = TaubinSmoother(
            iterations=settings.smoother_iterations,
            lamb=settings.smoother_lamb,
            nu=settings.smoother_nu,
        )
    else:
        smoother = LaplacianSmoother(
            iterations=settings.smoother_iterations,
            lamb=settings.smoother_lamb,
        )

    exporter = Stage1Exporter(
        settings=settings,
        ese_config=ESEConfig(
            n_electrodes=settings.n_electrodes,
            ese_offset_mm=settings.ese_offset_mm,
            ese_reference=settings.ese_reference,
        ),
        fiducials=fiducials,
        skip_fiducials=resolved_skip_fiducials,
    )

    pipeline = Stage1Pipeline(
        loader=loader,
        segmenter=segmenter,
        cleaner=cleaner,
        smoother=smoother,
        exporter=exporter,
    )
    return pipeline.run(
        resolved_path,
        output_dir=resolved_output_dir,
        closing_radius=settings.closing_radius,
        threshold=settings.threshold,
    )


def _load_or_require_fiducials(
    fiducials_path: str | Path | None, skip_fiducials: bool
) -> list[Fiducial]:
    if skip_fiducials:
        return []
    if fiducials_path is None:
        raise ValueError(
            "No fiducials provided. Pass --fiducials_path (or set VIRDA_FIDUCIALS_PATH), "
            "or use --skip_fiducials to run without fiducial-dependent steps."
        )
    fiducials = load_fiducials(Path(fiducials_path))
    if not fiducials:
        raise ValueError(f"No fiducials found in {fiducials_path}")
    return fiducials


_BOOL_VALUES = {"true", "false", "True", "False", "1", "0"}


def _normalize_cli_flags() -> None:
    """Allow `--skip-fiducials` (and `--skip_fiducials`) as a bare boolean flag."""
    args = sys.argv[1:]
    for flag in ("--skip-fiducials", "--skip_fiducials"):
        if flag not in args:
            continue
        index = args.index(flag)
        has_value = index + 1 < len(args) and args[index + 1] in _BOOL_VALUES
        value = args[index + 1] if has_value else "true"
        drop = 2 if has_value else 1
        sys.argv = ["virda"] + args[:index] + ["--skip_fiducials", value] + args[index + drop :]
        return


if __name__ == "__main__":
    _normalize_cli_flags()
    result = run()
    print(f"Stage 1 done. Mesh: {result.mesh.vertices.shape[0]} vertices")
