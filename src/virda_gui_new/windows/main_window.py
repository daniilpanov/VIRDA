from pathlib import Path
from typing import Any

from PySide6.QtWidgets import QMainWindow

from virda_gui_new.widgets.viewer3d import AdvancedViewer3D, MeshWithPoints


def _project_scene_sources(project: Path) -> dict[str, Any]:
    sources: dict[str, Any] = {}

    input_dir = project / "input"
    for pattern in ("*.nii.gz", "*.nii"):
        found = sorted(input_dir.glob(pattern))
        if found:
            sources["nifti"] = found[0]
            break

    fiducials = input_dir / "fiducials.json"
    if fiducials.is_file():
        sources["fiducials"] = fiducials

    scalp_mesh = project / "mesh" / "final_mesh.ply"
    if scalp_mesh.is_file():
        scalp_points = project / "localization" / "electrodes_scalp.json"
        sources["scalp"] = MeshWithPoints(
            scalp_mesh, scalp_points if scalp_points.is_file() else None
        )

    ese_mesh = project / "ese" / "ese_mesh.ply"
    if ese_mesh.is_file():
        ese_points = project / "localization" / "electrodes_ese.json"
        sources["ese"] = MeshWithPoints(ese_mesh, ese_points if ese_points.is_file() else None)

    return sources


class MainWindow(QMainWindow):
    def __init__(self, project_path: Path) -> None:
        super().__init__()

        self.setWindowTitle("VIRDA: Electrode Localization System")
        self._project_path = project_path

        self._viewer3d = AdvancedViewer3D(self, log=self._log)
        self.setCentralWidget(self._viewer3d)

        self.showMaximized()
        self.maximumSize()

        self.load_viewer_scene(project_path)

    def load_viewer_scene(self, project: Path) -> None:
        """Show the patient project's NIfTI, fiducials, scalp and ESE meshes."""
        sources = _project_scene_sources(project)
        if not sources:
            self._log("Nothing to view: no NIfTI or mesh files found in the project.")
            return
        self._viewer3d.set_scene(**sources)

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        self._viewer3d.shutdown()
        super().closeEvent(event)

    def _log(self, message: str) -> None:
        self.statusBar().showMessage(message)
