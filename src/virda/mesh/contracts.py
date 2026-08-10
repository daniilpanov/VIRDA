from typing import Protocol, runtime_checkable

import numpy as np

from virda.models.scalp_mesh import ScalpMesh


@runtime_checkable
class MeshCleaner(Protocol):
    def clean(
        self,
        mesh: ScalpMesh,
        *,
        mask: np.ndarray | None = None,
        affine: np.ndarray | None = None,
    ) -> ScalpMesh: ...


@runtime_checkable
class MeshSmoother(Protocol):
    def smooth(self, mesh: ScalpMesh) -> ScalpMesh: ...


@runtime_checkable
class MeshExtractor(Protocol):
    def extract(self, mask: np.ndarray, affine: np.ndarray) -> ScalpMesh: ...
