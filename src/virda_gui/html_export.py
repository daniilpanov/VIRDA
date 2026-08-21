"""Export a self-contained HTML viewer for a Stage 1 patient project.

``virda-gui-html`` reads a patient project (scalp mesh, fiducials)
and writes a single ``.html`` file that renders the scalp mesh in the browser
using three.js (loaded from a CDN). Mesh and fiducials are embedded in the
file, so the viewer needs no server and runs on desktop and mobile alike.
MRI volume rendering is currently disabled.

The scene placement matches ``virda_gui.viewer`` exactly: an axis-aligned
affine keeps world millimeters (the volume box is placed at the affine origin
with its spacing), any other affine moves the mesh and fiducials into voxel
index space. Both cases reuse the helpers from ``virda_gui.scene``.

Usage
-----
    virda-gui-html patient_project/CTRL_0001 --output viewer.html
    virda-gui-html patient_project/CTRL_0001 --output viewer.html --max-dim 96
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from virda_gui.scene import (
    compute_normal_lines,
    load_fiducial_points,
    load_normals,
    sample_normals,
    transform_points,
)

_DEFAULT_MAX_DIM = 128
_STEP_SIZE_VOXELS = 0.8

_DEBUG_UI = """
  <label><input type="checkbox" id="cb-nsteps"> Debug nsteps</label>
  <label><input type="checkbox" id="cb-nearest"> Nearest filter</label>
  <label>Step size <input type="range" id="r-step" min="0.1" max="2.0" step="0.05"
    value="0.8" style="width: 110px; vertical-align: middle;"></label>
"""

_DEBUG_JS = r"""
if (vol) {
(function () {
  function dbgLog(obj) { console.log('[virda-debug]', obj); }
  try {
    const gl = renderer.getContext();
    const dbgInfo = gl.getExtension('WEBGL_debug_renderer_info');
    dbgLog({
      three: 'r' + THREE.REVISION,
      webgl2: renderer.capabilities.isWebGL2,
      gpu: (dbgInfo && gl.getParameter(dbgInfo.UNMASKED_RENDERER_WEBGL))
        || gl.getParameter(gl.RENDERER),
      maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
      depthBits: gl.getParameter(gl.DEPTH_BITS),
      floatLinear: !!gl.getExtension('OES_texture_float_linear'),
      halfFloatLinear: !!gl.getExtension('OES_texture_half_float_linear'),
      volumeType: volumeTexture.type === THREE.HalfFloatType ? 'HalfFloat'
        : (volumeTexture.type === THREE.FloatType ? 'Float' : volumeTexture.type),
      volumeFilter: volumeTexture.minFilter === THREE.LinearFilter ? 'Linear' : 'Nearest',
      dims: [vol.dims[0], vol.dims[1], vol.dims[2]],
      spacing: vol.spacing,
      stepWorld: +stepWorld.toFixed(4),
      stepVox: +(stepWorld / minSpacing).toFixed(4),
      maxSteps: maxSteps,
      cameraPos: [camera.position.x, camera.position.y, camera.position.z]
        .map((v) => +v.toFixed(1)),
      cameraNear: camera.near,
      cameraFar: camera.far,
      cameraFov: camera.fov,
      climBase: vol.clim_base,
      climBoost: vol.clim_boost,
    });
  } catch (err) {
    dbgLog('diagnostic error: ' + err);
  }

  const cbNsteps = document.getElementById('cb-nsteps');
  const cbNearest = document.getElementById('cb-nearest');
  const rStep = document.getElementById('r-step');

  cbNsteps.addEventListener('change', () => {
    volumeMat.uniforms.u_debug.value = cbNsteps.checked ? 1.0 : 0.0;
    dbgLog('nsteps view ' + (cbNsteps.checked ? 'ON' : 'OFF'));
  });
  cbNearest.addEventListener('change', () => {
    volumeTexture.minFilter = cbNearest.checked ? THREE.NearestFilter : THREE.LinearFilter;
    volumeTexture.magFilter = cbNearest.checked ? THREE.NearestFilter : THREE.LinearFilter;
    volumeTexture.needsUpdate = true;
    dbgLog('voxel filter ' + (cbNearest.checked ? 'Nearest' : 'Linear'));
  });
  rStep.addEventListener('input', () => {
    const v = parseFloat(rStep.value);
    volumeMat.uniforms.u_rel_step.value = v;
    volumeMat.uniforms.u_alpha.value =
      stepWorld * (v / 0.8) / (cbBoost.checked ? 0.5 : 1.0);
    dbgLog('step size ' + v.toFixed(2) + ' vox, maxSteps ' + maxSteps
      + ', u_alpha ' + volumeMat.uniforms.u_alpha.value.toFixed(3));
  });
  dbgLog('READY: try Debug nsteps checkbox, Nearest filter, Step size slider');
})();
} /* end if (vol) */
"""

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1,
  maximum-scale=1, user-scalable=no">
<title>VIRDA — __DATASET__ — scalp mesh over MRI</title>
<style>
  html, body { margin: 0; height: 100%; overflow: hidden; background: #101018; color: #e8e8f0;
    font-family: system-ui, -apple-system, sans-serif; }
  #view { position: fixed; inset: 0; }
  #view canvas { display: block; }
  #ui { position: fixed; top: 10px; left: 10px; z-index: 10; background: rgba(16, 16, 24, 0.82);
    border: 1px solid #3a3a4a; border-radius: 8px; padding: 10px 12px; font-size: 13px;
    line-height: 1.9; user-select: none; }
  #ui label { display: block; cursor: pointer; }
  #ui input { margin-right: 8px; accent-color: #fa8072; }
  #labels { position: fixed; inset: 0; pointer-events: none; z-index: 5; }
  .fid-label { position: absolute; transform: translate(-50%, -160%); padding: 1px 6px;
    border-radius: 4px; background: rgba(0, 0, 0, 0.65); color: #ffd7d7; font-size: 12px;
    white-space: nowrap; }
  .axis-label { position: absolute; transform: translate(-50%, -50%); font-size: 15px;
    font-weight: 600; text-shadow: 0 0 6px #000; }
  #hint { position: fixed; bottom: 8px; left: 50%; transform: translateX(-50%); z-index: 10;
    color: #9a9aae; font-size: 12px; background: rgba(16, 16, 24, 0.7); padding: 4px 10px;
    border-radius: 12px; white-space: nowrap; }
  #loader { position: fixed; inset: 0; z-index: 100; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 14px; background: rgba(16, 16, 24, 0.92);
    color: #e8e8f0; font-size: 14px; }
  #loader .spinner { width: 44px; height: 44px; border: 4px solid rgba(250, 128, 114, 0.25);
    border-top-color: #fa8072; border-radius: 50%; animation: spin 0.9s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div id="view"></div>
<div id="labels"></div>
<div id="ui">
  <label><input type="checkbox" id="cb-mesh" checked> Show mesh</label>
  <!-- MRI volume rendering does not work properly -->
  <label><input type="checkbox" id="cb-mri" disabled> Show MRI</label>
  <label><input type="checkbox" id="cb-fid" checked> Show fiducials</label>
  <label><input type="checkbox" id="cb-elec"> Show electrodes</label>
  <label><input type="checkbox" id="cb-normals"> Show normals</label>
  <label><input type="checkbox" id="cb-boost"> Boost contrast</label>__DEBUG_UI__
  <div id="ese-info" style="display:none; margin-top:6px; padding-top:6px;
    border-top:1px solid #3a3a4a; font-size:12px; line-height:1.7;
    color:#b0b0c0;"></div>
</div>
<div id="hint">drag to rotate · wheel / pinch to zoom · shift-drag to pan</div>
<div id="loader"><div class="spinner"></div><div>Loading data…</div></div>
<script type="application/json" id="virda-data">__VIRDA_DATA__</script>
<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.170.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.170.0/examples/jsm/"
  }
}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const payload = JSON.parse(document.getElementById('virda-data').textContent);
const vol = payload.volume;
const meshData = payload.mesh;
const fids = payload.fiducials;

function decodeBytes(b64) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8;
}

function decodeUint16(b64) {
  const u8 = decodeBytes(b64);
  const out = new Uint16Array(u8.length / 2);
  const view = new DataView(u8.buffer);
  for (let i = 0; i < out.length; i++) out[i] = view.getUint16(i * 2, true);
  return out;
}

function decodeFloat32(b64) {
  const u8 = decodeBytes(b64);
  return new Float32Array(u8.buffer);
}

function decodeUint32(b64) {
  const u8 = decodeBytes(b64);
  return new Uint32Array(u8.buffer);
}

const renderer = new THREE.WebGLRenderer({ antialias: true, premultipliedAlpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x101018);
document.getElementById('view').appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 20000);

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const sun = new THREE.DirectionalLight(0xffffff, 0.9);
sun.position.set(1, 1.5, 1);
scene.add(sun);
const fill = new THREE.DirectionalLight(0xffffff, 0.3);
fill.position.set(-1, -0.5, -1);
scene.add(fill);

let diagWorld = 1.0;
let minSpacing = 1.0;
let boxMin = new THREE.Vector3(0, 0, 0);
let boxMax = new THREE.Vector3(1, 1, 1);

if (vol) {
  const dims = new THREE.Vector3(vol.dims[0], vol.dims[1], vol.dims[2]);
  boxMin = new THREE.Vector3(vol.origin[0], vol.origin[1], vol.origin[2]);
  boxMax = new THREE.Vector3(
  vol.origin[0] + vol.dims[0] * vol.spacing[0],
  vol.origin[1] + vol.dims[1] * vol.spacing[1],
  vol.origin[2] + vol.dims[2] * vol.spacing[2],
);

const volumeTexture = new THREE.Data3DTexture(
  decodeUint16(vol.data), vol.dims[0], vol.dims[1], vol.dims[2]
);
volumeTexture.format = THREE.RedFormat;
volumeTexture.type = THREE.HalfFloatType;
volumeTexture.minFilter = THREE.LinearFilter;
volumeTexture.magFilter = THREE.LinearFilter;
volumeTexture.wrapS = volumeTexture.wrapT = volumeTexture.wrapR = THREE.ClampToEdgeWrapping;
volumeTexture.flipY = false;
volumeTexture.generateMipmaps = false;
volumeTexture.unpackAlignment = 1;
volumeTexture.needsUpdate = true;

minSpacing = Math.min(vol.spacing[0], vol.spacing[1], vol.spacing[2]);
let stepWorld = __STEP_VOX__ * minSpacing;
diagWorld = boxMax.clone().sub(boxMin).length();
let maxSteps = Math.max(32, Math.min(512, Math.ceil(diagWorld / stepWorld)));

const VERT = `
  varying vec4 v_nearpos;
  varying vec4 v_farpos;
  varying vec3 v_position;
  void main() {
    vec4 position4 = vec4(position, 1.0);
    vec4 pos_in_cam = modelViewMatrix * position4;
    pos_in_cam.z = -pos_in_cam.w;
    v_nearpos = inverse(modelViewMatrix) * pos_in_cam;
    pos_in_cam.z = pos_in_cam.w;
    v_farpos = inverse(modelViewMatrix) * pos_in_cam;
    v_position = position;
    gl_Position = projectionMatrix * viewMatrix * modelMatrix * position4;
  }
`;

const FRAG = `
  precision highp float;
  precision mediump sampler3D;

  uniform sampler3D u_volume;
  uniform vec3 u_size;
  uniform vec2 u_clim;
  uniform float u_k;
  uniform float u_alpha;
  uniform float u_rel_step;
  uniform float u_debug;

  varying vec3 v_position;
  varying vec4 v_nearpos;
  varying vec4 v_farpos;

  vec3 boneLut(float t) {
    vec3 c0 = vec3(0.0, 0.0, 0.0);
    vec3 c1 = vec3(0.16, 0.16, 0.24);
    vec3 c2 = vec3(0.53, 0.53, 0.62);
    vec3 c3 = vec3(0.80, 0.80, 0.85);
    vec3 c4 = vec3(1.0, 1.0, 1.0);
    t = clamp(t, 0.0, 1.0);
    if (t < 0.25) return mix(c0, c1, t * 4.0);
    if (t < 0.5) return mix(c1, c2, (t - 0.25) * 4.0);
    if (t < 0.75) return mix(c2, c3, (t - 0.5) * 4.0);
    return mix(c3, c4, (t - 0.75) * 4.0);
  }

  void main() {
    vec3 farpos = v_farpos.xyz / v_farpos.w;
    vec3 nearpos = v_nearpos.xyz / v_nearpos.w;
    vec3 view_ray = normalize(nearpos - farpos);

    float distance = dot(nearpos - v_position, view_ray);
    distance = max(distance, min(
      (0.0 - v_position.x) / view_ray.x, (u_size.x - v_position.x) / view_ray.x));
    distance = max(distance, min(
      (0.0 - v_position.y) / view_ray.y, (u_size.y - v_position.y) / view_ray.y));
    distance = max(distance, min(
      (0.0 - v_position.z) / view_ray.z, (u_size.z - v_position.z) / view_ray.z));

    vec3 front = v_position + view_ray * distance;
    int nsteps = int(-distance / u_rel_step + 0.5);
    nsteps = min(nsteps, MAX_STEPS);
    if (nsteps < 1) discard;

    vec3 step = ((v_position - front) / u_size) / float(nsteps);
    vec3 loc = front / u_size;

    vec3 acc = vec3(0.0);
    float a = 0.0;
    for (int i = 0; i < MAX_STEPS; i++) {
      if (i >= nsteps) break;
      float v = texture(u_volume, loc).r;
      float x = (v - u_clim.x) / max(u_clim.y - u_clim.x, 1e-6);
      float s = clamp(x, 0.0, 1.0);
      float sig = 1.0 / (1.0 + exp(-u_k * (2.0 * s - 1.0)));
      float alpha = 1.0 - pow(1.0 - sig, u_alpha);
      if (alpha > 0.002) {
        acc += (1.0 - a) * alpha * boneLut(s);
        a += (1.0 - a) * alpha;
        if (a > 0.98) break;
      }
      loc += step;
    }
    if (u_debug > 0.5) {
      gl_FragColor = vec4(0.0, float(nsteps) / float(MAX_STEPS), 1.0, 1.0);
      return;
    }
    gl_FragColor = vec4(acc, a);
  }
`;

const volumeMat = new THREE.ShaderMaterial({
  uniforms: {
    u_volume: { value: volumeTexture },
    u_size: { value: dims },
    u_clim: { value: new THREE.Vector2(vol.clim_base[0], vol.clim_base[1]) },
    u_k: { value: 10.0 },
    u_alpha: { value: stepWorld },
    u_rel_step: { value: __STEP_VOX__ },
    u_debug: { value: 0.0 },
  },
  vertexShader: '#define STEP_VOX ' + __STEP_VOX__.toFixed(4) + '\n' + VERT,
  fragmentShader: '\n#define MAX_STEPS ' + maxSteps + '\n' + FRAG,
  transparent: true,
  depthWrite: false,
  side: THREE.BackSide,
});

const volume = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), volumeMat);
volume.geometry.translate(0.5, 0.5, 0.5);
volume.geometry.scale(vol.dims[0], vol.dims[1], vol.dims[2]);
volume.scale.set(vol.spacing[0], vol.spacing[1], vol.spacing[2]);
volume.position.set(vol.origin[0], vol.origin[1], vol.origin[2]);
volume.frustumCulled = false;
volume.renderOrder = 0;
scene.add(volume);
} /* end if (vol) */

let mesh = null;
let meshMat = null;
let meshGeo = null;
if (meshData && meshData.vertices && meshData.faces) {
  const verts = decodeFloat32(meshData.vertices);
  const faces = decodeUint32(meshData.faces);
  meshGeo = new THREE.BufferGeometry();
  meshGeo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  meshGeo.setIndex(new THREE.BufferAttribute(faces, 1));
  meshGeo.computeVertexNormals();
  meshMat = new THREE.MeshPhongMaterial({
    color: 0xfa8072,
    transparent: true,
    opacity: 0.6,
    side: THREE.DoubleSide,
    depthWrite: true,
    specular: 0x222222,
    shininess: 20,
  });
  mesh = new THREE.Mesh(meshGeo, meshMat);
  mesh.renderOrder = 1;
  scene.add(mesh);
}

if (mesh) {
  meshGeo.computeBoundingBox();
  const mb = meshGeo.boundingBox;
  boxMin.copy(mb.min);
  boxMax.copy(mb.max);
  diagWorld = mb.max.clone().sub(mb.min).length();
  const md = mb.max.clone().sub(mb.min);
  minSpacing = Math.min(md.x, md.y, md.z) || 1.0;
}

const fidGroup = new THREE.Group();
const fidLabels = fids && fids.points ? fids.points : [];
let fidMat = null;
if (fidLabels.length > 0) {
  const sphereR = diagWorld * 0.008;
  const geo = new THREE.SphereGeometry(Math.max(sphereR, 0.5), 12, 12);
  fidMat = new THREE.MeshBasicMaterial({ color: 0xff4040 });
  for (const p of fidLabels) {
    const s = new THREE.Mesh(geo, fidMat);
    s.position.set(p[0], p[1], p[2]);
    fidGroup.add(s);
  }
  scene.add(fidGroup);
}

const elecData = payload.electrodes;
const elecGroup = new THREE.Group();
const elecLinksGroup = new THREE.Group();
let elecLabelsData = [];
if (elecData && elecData.points && elecData.points.length > 0) {
  const sphereR = diagWorld * 0.006;
  const geo = new THREE.SphereGeometry(Math.max(sphereR, 0.4), 10, 10);
  const residuals = elecData.residuals || [];
  const flags = elecData.flags || [];
  const hasResiduals = residuals.some(r => r > 0);
  function jetColor(t) {
    t = Math.max(0, Math.min(1, t));
    const r = Math.min(1, Math.max(0, 1.5 - Math.abs(t - 0.75) * 4));
    const g = Math.min(1, Math.max(0, 1.5 - Math.abs(t - 0.5) * 4));
    const b = Math.min(1, Math.max(0, 1.5 - Math.abs(t - 0.25) * 4));
    return new THREE.Color(r, g, b);
  }
  let maxRes = 1;
  if (hasResiduals) {
    maxRes = Math.max(...residuals.filter(r => r > 0), 1);
  }
  const fidPosMap = {};
  if (fids && fids.points && fids.labels) {
    fids.labels.forEach((l, i) => { fidPosMap[l] = fids.points[i]; });
  }
  for (let i = 0; i < elecData.points.length; i++) {
    const p = elecData.points[i];
    const flagged = flags[i];
    const res = residuals[i] || 0;
    let color;
    if (flagged) {
      color = new THREE.Color(0xff2020);
    } else if (hasResiduals && res > 0) {
      color = jetColor(res / maxRes);
    } else {
      color = new THREE.Color(0xffff00);
    }
    const mat = new THREE.MeshBasicMaterial({ color });
    const s = new THREE.Mesh(geo, mat);
    s.position.set(p[0], p[1], p[2]);
    elecGroup.add(s);
    const eid = elecData.ids ? elecData.ids[i] : 'E' + (i + 1);
    elecLabelsData.push({ pos: p, id: eid, res, flagged });
    if (elecData.links && elecData.links[i]) {
      const link = elecData.links[i];
      for (const [fid, dist] of Object.entries(link)) {
        if (dist == null || !fidPosMap[fid]) continue;
        const lineGeo = new THREE.BufferGeometry();
        const fv = fidPosMap[fid];
        const verts = new Float32Array([
          p[0], p[1], p[2], fv[0], fv[1], fv[2],
        ]);
        lineGeo.setAttribute('position',
          new THREE.BufferAttribute(verts, 3));
        const lineMat = new THREE.LineBasicMaterial({
          color: 0x888888, transparent: true, opacity: 0.4,
        });
        elecLinksGroup.add(new THREE.LineSegments(lineGeo, lineMat));
      }
    }
  }
  elecGroup.visible = false;
  elecLinksGroup.visible = false;
  scene.add(elecGroup);
  scene.add(elecLinksGroup);
}

const normalsData = payload.normals;
let normalsGroup = null;
if (normalsData && normalsData.lines) {
  const lineVerts = decodeFloat32(normalsData.lines);
  const nSeg = normalsData.count;
  const positions = new Float32Array(nSeg * 6);
  for (let i = 0; i < nSeg; i++) {
    positions[i * 6 + 0] = lineVerts[i * 6 + 0];
    positions[i * 6 + 1] = lineVerts[i * 6 + 1];
    positions[i * 6 + 2] = lineVerts[i * 6 + 2];
    positions[i * 6 + 3] = lineVerts[i * 6 + 3];
    positions[i * 6 + 4] = lineVerts[i * 6 + 4];
    positions[i * 6 + 5] = lineVerts[i * 6 + 5];
  }
  const nGeo = new THREE.BufferGeometry();
  nGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const nMat = new THREE.LineBasicMaterial({
    color: 0x00ffff, linewidth: 2, transparent: true, opacity: 0.8
  });
  normalsGroup = new THREE.LineSegments(nGeo, nMat);
  normalsGroup.visible = false;
  scene.add(normalsGroup);
}

const center = boxMin.clone().add(boxMax).multiplyScalar(0.5);
const radius = diagWorld * 0.6;
camera.position.set(center.x + radius, center.y + radius * 0.8, center.z + radius * 1.1);
camera.lookAt(center);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(center);
controls.enableDamping = true;
controls.update();

const axes = new THREE.AxesHelper(radius * 0.35);
axes.position.copy(center);
scene.add(axes);

const labelLayer = document.getElementById('labels');
const axisCodes = payload.axes;

function worldToScreen(world, camera) {
  const v = world.clone().project(camera);
  return {
    x: (v.x + 1) * 0.5 * window.innerWidth,
    y: (-v.y + 1) * 0.5 * window.innerHeight,
    visible: v.z > -1 && v.z < 1,
  };
}

function updateLabels() {
  labelLayer.textContent = '';
  if (fidGroup.visible && fids && fids.labels) {
    for (let i = 0; i < fidLabels.length; i++) {
      const p = new THREE.Vector3(fidLabels[i][0], fidLabels[i][1], fidLabels[i][2]);
      const scr = worldToScreen(p, camera);
      if (!scr.visible) continue;
      const el = document.createElement('div');
      el.className = 'fid-label';
      el.textContent = fids.labels[i];
      el.style.left = scr.x + 'px';
      el.style.top = scr.y + 'px';
      labelLayer.appendChild(el);
    }
  }
  if (elecGroup.visible && elecLabelsData.length > 0) {
    const fontSize = Math.max(8, Math.min(11, diagWorld * 0.002));
    for (const ed of elecLabelsData) {
      const p = new THREE.Vector3(ed.pos[0], ed.pos[1], ed.pos[2]);
      const scr = worldToScreen(p, camera);
      if (!scr.visible) continue;
      const el = document.createElement('div');
      el.className = 'fid-label';
      el.style.fontSize = fontSize + 'px';
      el.style.background = ed.flagged ? 'rgba(180,0,0,0.7)' : 'rgba(0,0,0,0.55)';
      el.style.color = ed.flagged ? '#ffa0a0' : '#d0d0ff';
      el.textContent = ed.id;
      el.style.left = scr.x + 'px';
      el.style.top = scr.y + 'px';
      labelLayer.appendChild(el);
    }
  }
  if (axisCodes) {
    const dirs = [
      [1, 0, 0, '#ff5555', axisCodes[0]],
      [0, 1, 0, '#55ff55', axisCodes[1]],
      [0, 0, 1, '#5599ff', axisCodes[2]],
    ];
    const arm = radius * 0.35;
    for (const [dx, dy, dz, color, code] of dirs) {
      if (!code) continue;
      const p = new THREE.Vector3(
        center.x + dx * arm, center.y + dy * arm, center.z + dz * arm
      );
      const scr = worldToScreen(p, camera);
      if (!scr.visible) continue;
      const el = document.createElement('div');
      el.className = 'axis-label';
      el.style.color = color;
      el.style.left = scr.x + 'px';
      el.style.top = scr.y + 'px';
      el.textContent = code;
      labelLayer.appendChild(el);
    }
  }
}

const cbMesh = document.getElementById('cb-mesh');
const cbMri = document.getElementById('cb-mri');
const cbFid = document.getElementById('cb-fid');
const cbElec = document.getElementById('cb-elec');
const cbNormals = document.getElementById('cb-normals');
const cbBoost = document.getElementById('cb-boost');

cbMesh.addEventListener('change', () => { if (mesh) mesh.visible = cbMesh.checked; });
cbFid.addEventListener('change', () => {
  fidGroup.visible = cbFid.checked;
  if (fidMat) fidMat.visible = cbFid.checked;
});
cbElec.addEventListener('change', () => {
  elecGroup.visible = cbElec.checked;
  elecLinksGroup.visible = cbElec.checked;
});
cbNormals.addEventListener('change', () => {
  if (normalsGroup) normalsGroup.visible = cbNormals.checked;
});

if (normalsGroup) { cbNormals.checked = true; normalsGroup.visible = true; }

const eseInfo = document.getElementById('ese-info');
const eseCfg = payload.ese_config;
if (eseCfg && eseCfg.ese) {
  const e = eseCfg.ese;
  const lines = [];
  if (e.n_electrodes != null) lines.push('Electrodes: ' + e.n_electrodes);
  if (e.ese_offset_mm != null) lines.push('Offset: ' + e.ese_offset_mm + ' mm');
  if (e.ese_reference) lines.push('Reference: ' + e.ese_reference);
  if (lines.length) {
    eseInfo.innerHTML = '<b>ESE config</b><br>' + lines.join('<br>');
    eseInfo.style.display = 'block';
  }
}
if (vol) {
  const stepWorld = __STEP_VOX__ * minSpacing;
  function setBoost(on) {
    const clim = on ? vol.clim_boost : vol.clim_base;
    volumeMat.uniforms.u_clim.value.set(clim[0], clim[1]);
    volumeMat.uniforms.u_k.value = 10.0;
    volumeMat.uniforms.u_alpha.value = stepWorld / (on ? 0.5 : 1.0);
    if (meshMat) {
      meshMat.opacity = on ? 0.95 : 0.6;
      meshMat.specular.set(on ? 0xffffff : 0x222222);
      meshMat.shininess = on ? 40 : 20;
    }
  }
  cbBoost.addEventListener('change', () => setBoost(cbBoost.checked));
} else {
  cbBoost.addEventListener('change', () => {
    if (meshMat) {
      meshMat.opacity = cbBoost.checked ? 0.95 : 0.6;
      meshMat.specular.set(cbBoost.checked ? 0xffffff : 0x222222);
      meshMat.shininess = cbBoost.checked ? 40 : 20;
    }
  });
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

renderer.render(scene, camera);
document.getElementById('loader').style.display = 'none';

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  updateLabels();
__DEBUG_JS__
renderer.render(scene, camera);
}
animate();
</script>
</body>
</html>
"""


def _default_stride(dims: tuple[int, ...] | np.ndarray, max_dim: int) -> int:
    """Smallest voxel stride keeping the longest axis within ``max_dim``."""
    longest = int(np.max(np.asarray(dims, dtype=np.int64)))
    if longest <= max_dim:
        return 1
    return int(math.ceil(longest / max_dim))


def _encode_float16(data: np.ndarray) -> str:
    """Encode an array as little-endian half floats in base64."""
    packed = np.asarray(data, dtype="<f2").tobytes()
    return base64.b64encode(packed).decode("ascii")


def _encode_float32(points: np.ndarray) -> str:
    """Encode an (N, 3) array as little-endian single-precision floats."""
    packed = np.asarray(points, dtype="<f4").tobytes()
    return base64.b64encode(packed).decode("ascii")


def _encode_uint32(faces: np.ndarray) -> str:
    """Encode an (M, 3) array of face indices as little-endian uints."""
    packed = np.asarray(faces, dtype="<u4").tobytes()
    return base64.b64encode(packed).decode("ascii")


def build_payload(
    project_dir: str | Path,
    max_dim: int = _DEFAULT_MAX_DIM,
    normals_path: str | Path | None = None,
    normals_scale: float = 3.0,
    normals_density: int = 500,
    electrodes_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read a patient project and return the embedded-viewer payload dict.

    When a ``ese/`` directory exists with ESE mesh and normals, they are
    loaded automatically.  The ESE config is read from ``config/ese.json``
    when present.
    """
    project = Path(project_dir)

    # MRI volume rendering is disabled.
    # img = nib.load(_find_nifti(project))
    # data = img.get_fdata(dtype=np.float32)
    # if data.ndim == 4:
    #     data = data[..., 0]
    # affine = img.affine
    # stride = _default_stride(data.shape, max_dim)
    # if stride > 1:
    #     data, affine = downsample(data, affine, stride)
    # spacing, origin, transform, _ = scene_placement(affine)
    # lo, hi = percentile_clim(data)
    # vrange = max(hi - lo, 1e-12)
    # normalized = (np.clip(data, lo, hi) - lo) / vrange
    transform = np.eye(4)

    payload: dict[str, Any] = {
        "dataset": project.name,
    }

    # --- ESE config ----------------------------------------------------------
    ese_config_path = project / "config" / "ese.json"
    if ese_config_path.is_file():
        payload["ese_config"] = json.loads(ese_config_path.read_text(encoding="utf-8"))

    # --- Mesh ----------------------------------------------------------------
    vertices_path = project / "mesh" / "scalp_vertices.npy"
    faces_path = project / "mesh" / "scalp_faces.npy"
    if vertices_path.is_file() and faces_path.is_file():
        vertices = np.load(vertices_path)
        faces = np.load(faces_path)
        payload["mesh"] = {
            "vertices": _encode_float32(transform_points(vertices, transform)),
            "faces": _encode_uint32(faces),
        }

    # --- Fiducials -----------------------------------------------------------
    fiducials_path = project / "fiducials" / "fiducials.json"
    if fiducials_path.is_file():
        points, labels = load_fiducial_points(fiducials_path)
        if points.size:
            payload["fiducials"] = {
                "points": transform_points(points, transform).tolist(),
                "labels": labels,
            }

    # --- Normals -------------------------------------------------------------
    # Explicit --normals takes priority; otherwise auto-detect ese/normals.npy.
    normals_file: Path | None = None
    if normals_path is not None:
        candidate = Path(normals_path)
        if candidate.is_file():
            normals_file = candidate
    elif (project / "ese" / "normals.npy").is_file():
        normals_file = project / "ese" / "normals.npy"

    if normals_file is not None and vertices is not None:
        normals = load_normals(normals_file)
        idx, sampled = sample_normals(normals, normals_density)
        origins, tips = compute_normal_lines(
            transform_points(vertices, transform)[idx],
            sampled,
            normals_scale,
        )
        n = len(idx)
        lines = np.empty((n * 2, 3), dtype=np.float32)
        lines[0::2] = origins
        lines[1::2] = tips
        payload["normals"] = {
            "lines": _encode_float32(lines),
            "count": n,
        }

    # --- Electrodes -----------------------------------------------------------
    electrodes_file: Path | None = None
    if electrodes_path is not None:
        candidate = Path(electrodes_path)
        if candidate.is_file():
            electrodes_file = candidate
    elif (project / "localization" / "electrodes.json").is_file():
        electrodes_file = project / "localization" / "electrodes.json"

    if electrodes_file is not None:
        electrodes_raw = json.loads(electrodes_file.read_text(encoding="utf-8"))
        if isinstance(electrodes_raw, dict):
            electrodes_raw = electrodes_raw.get("electrodes", [])
        points = []
        residuals = []
        flags = []
        ids = []
        links = []
        for item in electrodes_raw:
            coords = item.get("coords") or item.get("scalp_coords")
            if coords is None:
                continue
            points.append(
                transform_points(np.asarray(coords)[np.newaxis, :], transform)[0].tolist()
            )
            residuals.append(float(item.get("residual_error") or 0.0))
            flags.append(bool(item.get("flagged", False)))
            ids.append(item.get("electrode_id", ""))
            links.append({str(k): float(v) for k, v in item.get("measured_distances", {}).items()})
        if points:
            payload["electrodes"] = {
                "points": points,
                "residuals": residuals,
                "flags": flags,
                "ids": ids,
                "links": links,
            }

    return payload


def render_html(payload: dict[str, Any], debug: bool = False) -> str:
    """Render the self-contained HTML page for a payload dict."""
    dataset = str(payload.get("dataset", "patient"))
    data_json = json.dumps(payload).replace("</", "<\\/")
    debug_ui = _DEBUG_UI if debug else ""
    debug_js = _DEBUG_JS if debug else ""
    html = (
        _HTML_TEMPLATE.replace("__DATASET__", dataset)
        .replace("__VIRDA_DATA__", data_json)
        .replace("__DEBUG_UI__", debug_ui)
        .replace("__DEBUG_JS__", debug_js)
        .replace("__STEP_VOX__", f"{_STEP_SIZE_VOXELS:.4f}")
    )
    return html


def export_project(
    project_dir: str | Path,
    output: str | Path,
    max_dim: int = _DEFAULT_MAX_DIM,
    debug: bool = False,
    normals_path: str | Path | None = None,
    normals_scale: float = 3.0,
    normals_density: int = 500,
    electrodes_path: str | Path | None = None,
) -> Path:
    """Write the self-contained HTML viewer for a patient project."""
    payload = build_payload(
        project_dir,
        max_dim=max_dim,
        normals_path=normals_path,
        normals_scale=normals_scale,
        normals_density=normals_density,
        electrodes_path=electrodes_path,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(payload, debug=debug), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="virda-gui-html",
        description="Export a self-contained HTML viewer for a patient project.",
    )
    parser.add_argument("project_dir", help="Path to the patient project directory.")
    parser.add_argument(
        "-o", "--output", default="viewer.html", help="Output HTML file (default: viewer.html)."
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=_DEFAULT_MAX_DIM,
        help="Longest volume axis kept after downsampling (default: 128).",
    )
    parser.add_argument(
        "--normals",
        help="Path to normals file (normals.npy) for visualisation.",
    )
    parser.add_argument(
        "--normals-scale",
        type=float,
        default=3.0,
        help="Visual length of normal arrows in scene units (default: 3.0).",
    )
    parser.add_argument(
        "--normals-density",
        type=int,
        default=500,
        help="Show one normal per N vertices (default: 500).",
    )
    parser.add_argument(
        "--electrodes",
        help=(
            "Path to electrodes JSON (localization/electrodes.json). Auto-detected in project dir."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Add nsteps/filter/step diagnostics controls to the viewer.",
    )
    args = parser.parse_args()

    export_project(
        args.project_dir,
        args.output,
        max_dim=args.max_dim,
        debug=args.debug,
        normals_path=args.normals,
        normals_scale=args.normals_scale,
        normals_density=args.normals_density,
        electrodes_path=args.electrodes,
    )
    print(f"HTML viewer written to {args.output}")


if __name__ == "__main__":
    main()
