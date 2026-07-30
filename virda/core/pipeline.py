"""Pipeline orchestrator for the full VIRDA workflow."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..api.exporter import (
    create_project_folder,
    export_config_json,
    export_electrodes_csv,
    export_electrodes_json,
    export_ese_pairs_csv,
    export_faces_csv,
    export_fiducials_json,
    export_mesh,
    export_normals_csv,
    export_vertices_csv,
)
from ..core.ese_config import ESEConfig
from ..core.ese_generator import generate_ese
from ..core.electrode_localizer import localize_electrodes
from ..core.fiducial_manager import FiducialManager
from ..core.head_segmenter import segment_head
from ..core.measurement_importer import MeasurementImporter
from ..core.mesh_cleaner import clean_mesh
from ..core.mri_loader import load_mri
from ..core.pca_normal_estimator import estimate_normals_pca
from ..core.quality_control import validate_stage1, validate_stage2, validate_stage3
from ..core.surface_extractor import extract_surface
from ..core.types import ESEResult, LocalizationResult, MeshData

logger = logging.getLogger(__name__)


class VIRDAPipeline:
    """Orchestrates the full Stage 1 -> Stage 2 -> Stage 3 pipeline.

    Parameters
    ----------
    config : ESEConfig, optional
        ESE configuration. Uses defaults if None.
    """

    def __init__(self, config: ESEConfig | None = None) -> None:
        self.config = config or ESEConfig()
        self._stage1_output: dict | None = None
        self._stage2_output: dict | None = None

    def run_stage1(
        self,
        mri_path: Path,
        output_dir: Path,
        fiducial_coords: dict[str, np.ndarray] | None = None,
        smooth_iterations: int = 0,
    ) -> dict:
        """Run Stage 1: MRI loading, segmentation, mesh generation.

        Parameters
        ----------
        mri_path : Path
            Path to MRI file or DICOM directory.
        output_dir : Path
            Directory for output files.
        fiducial_coords : dict[str, np.ndarray], optional
            Pre-defined fiducial coordinates {id: np.array([x,y,z])}.
        smooth_iterations : int
            Mesh smoothing iterations.

        Returns
        -------
        dict
            Stage 1 outputs: mri, segmentation, mesh, fiducial_mgr, stats.
        """
        folders = create_project_folder(output_dir)

        logger.info("Stage 1: Loading MRI from %s", mri_path)
        mri = load_mri(mri_path)

        logger.info("Stage 1: Segmenting head")
        seg = segment_head(mri)

        logger.info("Stage 1: Extracting surface mesh")
        mesh = extract_surface(seg.mask, mri.voxel_size, affine=mri.affine)

        logger.info("Stage 1: Cleaning mesh")
        mesh, clean_stats = clean_mesh(mesh, smooth_iterations=smooth_iterations)

        head_centroid = mesh.vertices.mean(axis=0)
        fiducial_mgr = FiducialManager(
            head_centroid=head_centroid,
            surface_vertices=mesh.vertices,
        )

        if fiducial_coords:
            names = {"NAS": "Nasion", "LPA": "Left Preauricular", "RPA": "Right Preauricular", "INI": "Inion"}
            for fid_id, coords in fiducial_coords.items():
                fiducial_mgr.add_fiducial(
                    fid_id, names.get(fid_id, fid_id), coords
                )

        qc_messages = validate_stage1(
            mri=mri,
            mesh=mesh,
            fiducial_coords=fiducial_mgr.get_coordinates_matrix() if fiducial_coords else None,
            ese_offset_mm=self.config.offset_mm,
        )
        for msg in qc_messages:
            logger.warning("Stage 1 QC: %s", msg)

        export_mesh(mesh, folders["mesh"] / "scalp.ply")
        export_vertices_csv(mesh, folders["mesh"] / "vertices.csv")
        export_faces_csv(mesh, folders["mesh"] / "faces.csv")
        if fiducial_coords:
            export_fiducials_json(fiducial_mgr, folders["fiducials"] / "fiducials.json")
        export_config_json(self.config, folders["config"] / "parameters.json")

        self._stage1_output = {
            "mri": mri,
            "segmentation": seg,
            "mesh": mesh,
            "fiducial_mgr": fiducial_mgr,
            "clean_stats": clean_stats,
            "qc_messages": qc_messages,
            "output_dir": output_dir,
        }

        logger.info("Stage 1 complete: %d vertices, %d faces", mesh.num_vertices, mesh.num_faces)
        return self._stage1_output

    def run_stage2(
        self,
        stage1_output: dict | None = None,
        output_dir: Path | None = None,
        radius_mm: float = 10.0,
        min_neighbors: int = 5,
    ) -> dict:
        """Run Stage 2: PCA normal estimation and ESE generation.

        Parameters
        ----------
        stage1_output : dict, optional
            Output from run_stage1(). Uses cached if None.
        output_dir : Path, optional
            Directory for output files. Uses stage1 output_dir if None.
        radius_mm : float
            PCA neighborhood radius in mm.
        min_neighbors : int
            Minimum neighbors for PCA.

        Returns
        -------
        dict
            Stage 2 outputs: normal_result, ese, qc_messages.
        """
        s1 = stage1_output or self._stage1_output
        if s1 is None:
            raise RuntimeError("Stage 1 must be run before Stage 2")
        out_dir = output_dir or s1["output_dir"]
        mesh: MeshData = s1["mesh"]

        folders = create_project_folder(out_dir)

        logger.info("Stage 2: Estimating PCA normals (radius=%.1f mm)", radius_mm)
        normal_result = estimate_normals_pca(mesh, radius_mm=radius_mm, min_neighbors=min_neighbors)

        logger.info("Stage 2: Generating ESE surface")
        ese = generate_ese(mesh, normal_result, self.config)

        qc_messages = validate_stage2(ese)
        for msg in qc_messages:
            logger.warning("Stage 2 QC: %s", msg)

        export_normals_csv(normal_result, folders["mesh"] / "normals.csv")
        export_ese_pairs_csv(ese, folders["mesh"] / "ese_pairs.csv")
        export_mesh(mesh, folders["mesh"] / "scalp_clean.ply")

        self._stage2_output = {
            "normal_result": normal_result,
            "ese": ese,
            "qc_messages": qc_messages,
            "output_dir": out_dir,
        }

        logger.info("Stage 2 complete: %d ESE points", ese.num_points)
        return self._stage2_output

    def run_stage3(
        self,
        measurements_path: Path,
        stage2_output: dict | None = None,
        output_dir: Path | None = None,
        max_residual_threshold: float = 5.0,
    ) -> dict:
        """Run Stage 3: Electrode localization.

        Parameters
        ----------
        measurements_path : Path
            Path to measurements CSV or JSON file.
        stage2_output : dict, optional
            Output from run_stage2(). Uses cached if None.
        output_dir : Path, optional
            Directory for output files.
        max_residual_threshold : float
            Maximum acceptable residual error per electrode.

        Returns
        -------
        dict
            Stage 3 outputs: localization_result, qc_messages.
        """
        s2 = stage2_output or self._stage2_output
        if s2 is None:
            raise RuntimeError("Stage 2 must be run before Stage 3")
        out_dir = output_dir or s2["output_dir"]
        s1 = self._stage1_output or {}
        fiducial_mgr: FiducialManager = s1.get("fiducial_mgr")
        if fiducial_mgr is None:
            raise RuntimeError("FiducialManager not available from Stage 1")
        ese: ESEResult = s2["ese"]

        folders = create_project_folder(out_dir)

        logger.info("Stage 3: Loading measurements from %s", measurements_path)
        measurements_path = Path(measurements_path)
        importer = MeasurementImporter(
            fiducial_ids=list(fiducial_mgr.get_all_fiducials().keys())
        )
        if measurements_path.suffix == ".csv":
            importer.import_csv(measurements_path)
        else:
            importer.import_json(measurements_path)

        logger.info("Stage 3: Localizing electrodes")
        result = localize_electrodes(
            ese=ese,
            fiducial_mgr=fiducial_mgr,
            measurements=importer.get_all_measurements(),
            max_residual_threshold=max_residual_threshold,
        )

        qc_messages = validate_stage3(result, max_residual_threshold)
        for msg in qc_messages:
            logger.warning("Stage 3 QC: %s", msg)

        export_electrodes_csv(result, folders["results"] / "electrodes.csv")
        export_electrodes_json(result, folders["results"] / "electrodes.json")

        logger.info("Stage 3 complete: %d electrodes localized", result.num_electrodes)
        return {
            "localization_result": result,
            "qc_messages": qc_messages,
            "output_dir": out_dir,
        }

    def run_all(
        self,
        mri_path: Path,
        measurements_path: Path,
        output_dir: Path,
        fiducial_coords: dict[str, np.ndarray] | None = None,
        radius_mm: float = 10.0,
        max_residual_threshold: float = 5.0,
    ) -> dict:
        """Run the full Stage 1 -> 2 -> 3 pipeline.

        Parameters
        ----------
        mri_path : Path
            Path to MRI data.
        measurements_path : Path
            Path to distance measurements.
        output_dir : Path
            Output directory.
        fiducial_coords : dict[str, np.ndarray], optional
            Pre-defined fiducial coordinates.
        radius_mm : float
            PCA neighborhood radius.
        max_residual_threshold : float
            Maximum residual error threshold.

        Returns
        -------
        dict
            Combined results from all stages.
        """
        s1 = self.run_stage1(mri_path, output_dir, fiducial_coords)
        s2 = self.run_stage2(s1, radius_mm=radius_mm)
        s3 = self.run_stage3(measurements_path, s2, max_residual_threshold=max_residual_threshold)
        return {"stage1": s1, "stage2": s2, "stage3": s3}
