import logging

import numpy as np
import trimesh

from virda.mesh.adjacency import build_scalp_mesh
from virda.mesh.contracts import MeshPostprocessor
from virda.models.scalp_mesh import ScalpMesh

logger = logging.getLogger(__name__)


class QuadraticDecimator(MeshPostprocessor):
    """Reduce mesh density by a percentage of the original vertex count.

    ``density_percent`` expresses how much of the mesh is kept: ``100`` is a
    no-op (full mesh as extracted), ``50`` removes roughly half the vertices,
    ``0`` would remove everything and is rejected.  The reduction is a
    quadric-metric edge-collapse decimation built on ``trimesh`` /
    ``fast-simplification``.
    """

    def __init__(self, density_percent: float = 100.0) -> None:
        if not 1.0 <= density_percent <= 100.0:
            raise ValueError(f"density_percent must be within [1, 100], got {density_percent}")
        self._density_percent = density_percent
        if density_percent < 100.0:
            try:
                import fast_simplification  # type: ignore[import-untyped]  # noqa: F401
            except ImportError:
                raise RuntimeError(
                    "Mesh density below 100% requires the 'fast-simplification' package: "
                    "install it with `uv add fast-simplification`"
                    " (or `pip install fast-simplification`)"
                ) from None

    def _process(self, mesh: ScalpMesh) -> ScalpMesh:
        if self._density_percent >= 100.0 or mesh.faces.shape[0] == 0:
            logger.info(
                "Mesh density 100%%: keeping the full mesh (%d vertices)",
                len(mesh.vertices),
            )
            return mesh

        target_faces = max(1, round(mesh.faces.shape[0] * self._density_percent / 100.0))
        if target_faces >= mesh.faces.shape[0]:
            return mesh

        trimesh_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        decimated = trimesh_mesh.simplify_quadric_decimation(face_count=target_faces)
        logger.info(
            "Mesh density %.1f%%: decimating %d -> %d vertices",
            self._density_percent,
            len(mesh.vertices),
            len(decimated.vertices),
        )
        return build_scalp_mesh(
            vertices=np.asarray(decimated.vertices, dtype=np.float64),
            faces=np.asarray(decimated.faces, dtype=np.int64),
        )