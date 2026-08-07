"""Quality-control artifacts: 2D slice overlays, 3D renders, interactive HTML."""

from pathlib import Path

from virda.io.qc.render import render_3d
from virda.io.qc.slices import overlay_slices
from virda.io.qc.viewer import write_viewer_html
from virda.models.stage1_result import Stage1Result

__all__ = ["render_3d", "overlay_slices", "write_viewer_html", "run_qc"]


def run_qc(result: Stage1Result, output_dir: str | Path, with_html: bool = False) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    overlay_slices(result, out)
    render_3d(result, out)
    if with_html:
        write_viewer_html(result, out)
    return out
