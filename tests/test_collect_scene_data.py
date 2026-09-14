"""Unit tests for :func:`virda_gui.viewer.collect_scene_data`.

``collect_scene_data`` performs all file IO and scene preparation off the GUI
thread and returns plain data objects (NumPy arrays, PyVista meshes), so these
tests need no display and no ``QApplication``.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import trimesh

from virda_gui.viewer import SceneData, collect_scene_data


def _write_tsv(path: Path, rows: list[tuple[str, float, float, float]]) -> None:
    lines = ["name\tx\ty\tz"]
    lines += [f"{name}\t{x}\t{y}\t{z}" for name, x, y, z in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sphere_mesh(path: Path, radius: float = 80.0) -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    mesh.export(str(path))


def _write_t1(path: Path, shape: tuple[int, int, int] = (24, 24, 24)) -> None:
    affine = np.eye(4)
    affine[:3, :3] = np.diag([2.0, 2.0, 2.0])
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    nib.save(nib.Nifti1Image(data, affine), str(path))


class TestCollectSceneDataValidation:
    def test_requires_nifti_or_mesh(self) -> None:
        with pytest.raises(ValueError, match="at least one of"):
            collect_scene_data()

    def test_cras_requires_nifti(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="electrodes_cras requires"):
            collect_scene_data(mesh_path=str(tmp_path / "missing.ply"), electrodes_cras=True)


class TestCollectSceneData:
    def test_mesh_only_scene(self, tmp_path: Path) -> None:
        mesh_path = tmp_path / "mesh.ply"
        _write_sphere_mesh(mesh_path)

        scene = collect_scene_data(mesh_path=str(mesh_path))

        assert isinstance(scene, SceneData)
        assert scene.volume is None
        assert scene.scene_mesh is not None
        assert scene.scene_mesh.n_points > 0
        assert scene.mm_scene is True

    def test_nifti_only_scene(self, tmp_path: Path) -> None:
        nifti_path = tmp_path / "t1.nii"
        _write_t1(nifti_path)

        scene = collect_scene_data(nifti_path=str(nifti_path))

        assert scene.volume is not None
        assert scene.hi_clim is not None
        assert scene.scene_mesh is None

    def test_full_scene_with_overlays(self, tmp_path: Path) -> None:
        nifti_path = tmp_path / "t1.nii"
        mesh_path = tmp_path / "mesh.ply"
        fiducials_path = tmp_path / "fiducials.json"
        normals_path = tmp_path / "normals.npy"
        electrodes_path = tmp_path / "electrodes.tsv"
        _write_t1(nifti_path)
        _write_sphere_mesh(mesh_path)
        fiducials_path.write_text(
            '{"fiducials": [{"fiducial_id": "Nz", "name": "Nasion", '
            '"coordinates": [0.0, 80.0, 0.0], "coordinate_system": "world", '
            '"definition_method": "manual"}]}',
            encoding="utf-8",
        )
        electrodes_path.write_text(
            "name\tx\ty\tz\nE1\t80.0\t0.0\t0.0\nE2\t0.0\t0.0\t80.0\n",
            encoding="utf-8",
        )
        # Per-vertex normals: preserve the vertex count for later rendering.
        mesh = trimesh.load(str(mesh_path), force="mesh")
        assert isinstance(mesh, trimesh.Trimesh)
        n_vertices = len(mesh.vertices)
        np.save(normals_path, np.zeros((n_vertices, 3)))

        scene = collect_scene_data(
            nifti_path=str(nifti_path),
            mesh_path=str(mesh_path),
            fiducials_path=str(fiducials_path),
            normals_path=str(normals_path),
            electrode_specs=[(str(electrodes_path), "yellow")],
        )

        assert scene.volume is not None
        assert scene.scene_mesh is not None
        assert scene.normals_poly is not None
        assert scene.fiducial_points is not None
        assert scene.fiducial_labels == ["Nz (Nasion)"]
        assert len(scene.electrode_groups) == 1
        group = scene.electrode_groups[0]
        assert group["label"] == "electrodes.tsv"
        assert group["color"] == "yellow"
        assert group["points"].shape == (2, 3)

    def test_electrode_group_palette_color(self, tmp_path: Path) -> None:
        mesh_path = tmp_path / "mesh.ply"
        electrodes_path = tmp_path / "electrodes.tsv"
        _write_sphere_mesh(mesh_path)
        _write_tsv(electrodes_path, [("E1", 80.0, 0.0, 0.0)])

        scene = collect_scene_data(
            mesh_path=str(mesh_path),
            electrode_specs=[(str(electrodes_path), None)],
        )

        assert scene.electrode_groups[0]["color"] == "yellow"
