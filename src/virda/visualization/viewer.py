"""QC interactive three.js HTML viewer (requires pyvista for decimation)."""

# ruff: noqa: E501 - long lines are inside the generated JS/HTML template

import json
from pathlib import Path

import numpy as np

from virda.geometry.transforms import fiducials_world_coordinates
from virda.models.stage1_result import Stage1Result


def write_viewer_html(
    result: Stage1Result, output_dir: str | Path, mesh_path: str | Path | None = None
) -> Path:
    """Self-contained HTML viewer (decimated mesh; needs internet for the three.js CDN)."""
    out = Path(output_dir)
    mesh_file = Path(mesh_path) if mesh_path is not None else out / "mesh.ply"
    try:
        import pyvista as pv
    except ImportError:
        return out
    pv.OFF_SCREEN = True
    mesh = pv.read(str(mesh_file))
    decimated = mesh.decimate_pro(0.85)
    vertices = decimated.points.astype(np.float32).ravel().tolist()
    faces = decimated.faces.reshape(-1, 4)[:, 1:].astype(np.int32).ravel().tolist()
    center = decimated.points.mean(0)
    fid_points = fiducials_world_coordinates(result.fiducials, result.mri_volume.affine)
    fid_spheres = (fid_points - center).astype(np.float32).ravel().tolist()

    fiducial_js = ""
    if fid_spheres:
        sprite_js = "\n".join(
            f"const s{i} = new THREE.Sprite(new THREE.SpriteMaterial({{color:0xffcc00, depthTest:false}}));"
            f"s{i}.scale.set(0.06,0.06,1); s{i}.position.set({x},{y},{z}); scene.add(s{i});"
            for i, (x, y, z) in enumerate(fid_points - center)
        )
        fiducial_js = f"""
const fpts = new Float32Array({json.dumps(fid_spheres)});
const fgeo = new THREE.BufferGeometry();
fgeo.setAttribute('position', new THREE.BufferAttribute(fpts, 3));
const fmat = new THREE.MeshStandardMaterial({{color:0xff3333, roughness:0.2, metalness:0.1}});
const fsphere = new THREE.Mesh(fgeo, fmat);
scene.add(fsphere);
{sprite_js}
"""

    source_name = str(result.mri_volume.metadata.get("source", "head")).split("/")[-1]
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Head mesh QC</title>
<style>body{{margin:0;overflow:hidden;background:#111;font-family:sans-serif}}
#info{{position:fixed;top:8px;left:12px;color:#ccc;font-size:13px;z-index:10}}</style>
</head><body><div id="info">{source_name} — scalp mesh (decimated) — drag to rotate, scroll to zoom</div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const verts = new Float32Array({json.dumps(vertices)});
const faces = new Uint32Array({json.dumps(faces)});
const geo = new THREE.BufferGeometry();
geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
geo.setIndex(new THREE.BufferAttribute(faces, 1));
geo.computeVertexNormals();
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);
const cam = new THREE.PerspectiveCamera(45, innerWidth/innerHeight, 0.01, 100);
const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({{color:0x9fb4cc, roughness:0.45, metalness:0.05}}));
scene.add(m);
{fiducial_js}
scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const d = new THREE.DirectionalLight(0xffffff, 0.9); d.position.set(1,2,1.5); scene.add(d);
const d2 = new THREE.DirectionalLight(0xffffff, 0.3); d2.position.set(-1,-1,-1); scene.add(d2);
const renderer = new THREE.WebGLRenderer({{antialias:true}});
renderer.setSize(innerWidth, innerHeight); renderer.setPixelRatio(devicePixelRatio);
document.body.appendChild(renderer.domElement);
const ctrl = new THREE.OrbitControls(cam, renderer.domElement);
cam.position.set(2.2, 0.6, 1.8); ctrl.target.set(0,0,0);
addEventListener('resize', ()=>{{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();renderer.setSize(innerWidth, innerHeight)}});
(function tick(){{requestAnimationFrame(tick); ctrl.update(); renderer.render(scene, cam)}})();
</script></body></html>"""
    (out / "head_viewer.html").write_text(html)
    return out
