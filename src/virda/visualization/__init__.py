"""Visualization artifacts: 2D slice overlays, 3D renders, interactive HTML."""

from pathlib import Path

from virda.models.stage1_result import Stage1Result
from virda.visualization.render import render_3d
from virda.visualization.slices import overlay_slices
from virda.visualization.viewer import write_viewer_html

__all__ = ["overlay_slices", "render_3d", "write_viewer_html", "write_visual_artifacts"]


def write_visual_artifacts(
    result: Stage1Result,
    output_dir: str | Path,
    mesh_path: str | Path | None = None,
    with_html: bool = False,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    overlay_slices(result, out)
    render_3d(result, out, mesh_path=mesh_path)
    if with_html:
        write_viewer_html(result, out, mesh_path=mesh_path)
    return out
