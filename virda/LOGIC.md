# VIRDA — Mathematical Logic & Algorithms

## Overview

VIRDA (VIrtual Reality electrode Discovery Algorithm) localizes EEG electrode positions on the scalp surface using MRI-derived head mesh and distance measurements from electrodes to anatomical fiducials. The pipeline has 3 stages:

```
MRI → [Stage 1] Segmentation + Mesh → [Stage 2] PCA Normals + ESE → [Stage 3] Electrode Localization
```

---

## Stage 1: MRI Segmentation & Mesh Extraction

### 1.1 Coordinate Systems

MRI data is defined in voxel coordinates `v = (i, j, k)` and mapped to world coordinates `p = (x, y, z)` via an affine transform `A ∈ ℝ^{4×4}`:

```
p_homogeneous = A · v_homogeneous
```

where `v_homogeneous = (i, j, k, 1)^T` and `p = p_homogeneous[:3]`.

Voxel spacing `s = (s_x, s_y, s_z)` is extracted from the affine diagonal:

```
s_i = ||A[:3, i]||₂
```

### 1.2 Gaussian Smoothing

Before thresholding, the MRI volume `I` is smoothed with a Gaussian kernel to reduce noise:

```
I_smooth(x, y, z) = (G_σ * I)(x, y, z)
```

where `G_σ(x, y, z) = (2πσ²)^{-3/2} · exp(-(x² + y² + z²) / (2σ²))` and `σ` is configurable (`smooth_sigma`, default 1.0).

### 1.3 Otsu Thresholding (ThresholdSegmenter)

Otsu's method finds the threshold `T` that maximizes inter-class variance between foreground (head) and background:

```
T* = argmax_T [ ω₀(T) · ω₁(T) · (μ₀(T) - μ₁(T))² ]
```

where:
- `ω₀(T) = Σ_{i=0}^{T} p(i)` — cumulative probability of class 0 (background)
- `ω₁(T) = 1 - ω₀(T)` — cumulative probability of class 1 (foreground)
- `μ₀(T) = Σ_{i=0}^{T} i·p(i) / ω₀(T)` — mean intensity of class 0
- `μ₁(T) = Σ_{i=T+1}^{L} i·p(i) / ω₁(T)` — mean intensity of class 1
- `p(i)` = normalized histogram of voxel intensities

The binary mask is:

```
M(x, y, z) = 1  if I_smooth(x, y, z) > T*
              0  otherwise
```

### 1.4 Region Growing (RegionGrowSegmenter)

Alternative segmentation. Threshold is set as:

```
T = μ_I + k · σ_I
```

where `μ_I` = mean intensity, `σ_I` = standard deviation, `k` = `threshold_sigma` (default 2.0).

### 1.5 Morphological Closing

After thresholding, morphological closing (dilation followed by erosion) fills small holes in the binary mask:

```
M_closed = (M ⊕ B) ⊖ B
```

where `B` is a ball structuring element of radius `close_radius` (default 3 voxels).

### 1.6 Connected Component Analysis

Connected components are labeled via flood-fill. Only components with `≥ min_component_size` voxels (default 10,000) are kept. The largest component is assumed to be the head.

### 1.7 Marching Cubes Surface Extraction

The binary mask is converted to a triangular mesh using the Marching Cubes algorithm (Lorensen & Cline, 1987):

- Isosurface level = 0.5 (binary boundary)
- Each voxel cube is classified into one of 256 cases based on which vertices are inside/outside
- Triangle vertices are interpolated along cube edges:

```
v_vertex = v₁ + (level - s₁) / (s₂ - s₁) · (v₂ - v₁)
```

where `s₁, s₂` are scalar values at vertices `v₁, v₂` of the edge.

Spacing between vertices = voxel spacing `s`.

If an affine is provided, vertices are transformed to world coordinates:

```
v_world = (A · [v_vertex; 1])[:3]
```

### 1.8 Vertex Normal Computation

Per-vertex normals are computed as the area-weighted average of adjacent face normals:

**Face normals** (cross product):

```
n_face = (v₁ - v₀) × (v₂ - v₀)
n_face = n_face / ||n_face||₂
```

**Vertex normals** (averaging):

```
n_vertex_i = Σ_{f ∈ F(i)} n_face_f
n_vertex_i = n_vertex_i / ||n_vertex_i||₂
```

where `F(i)` = set of faces containing vertex `i`.

### 1.9 Mesh Cleaning

#### Degenerate Face Removal

Faces where two or more vertices coincide are removed:

```
valid_face = (v₀ ≠ v₁) ∧ (v₁ ≠ v₂) ∧ (v₀ ≠ v₂)
```

#### Duplicate Vertex Merging

Vertices with identical coordinates are merged using `np.unique`. Face indices are remapped to the new vertex array.

#### Small Component Pruning

Connected components (via `trimesh.split`) with face count < `min_component_fraction × max_component_faces` are removed.

#### Winding Order Fix

`trimesh.fix_normals()` ensures all faces have consistent orientation (outward-facing normals).

#### Laplacian Smoothing

Iterative smoothing that moves each vertex toward the centroid of its neighbors:

```
v'_i = v_i + λ · (ē_i - v_i)
```

where:
- `ē_i = (1 / |N(i)|) · Σ_{j ∈ N(i)} v_j` — mean of neighbors
- `N(i)` = set of adjacent vertices (connected by an edge)
- `λ` = smoothing weight (`smooth_lambda`, default 0.5)
- Iterations controlled by `smooth_iterations` (default 0 = no smoothing)

In matrix form:

```
v' = v + λ · (L · v)
```

where `L = D⁻¹·A - I` is the Laplacian matrix, `A` is adjacency, `D` is degree matrix.

---

## Stage 2: PCA Normal Estimation & ESE Generation

### 2.1 Local PCA Normal Estimation

For each vertex `v_i`, a local neighborhood `N(i)` is found via radius search (`cKDTree.query_ball_point`) or k-NN.

#### Unweighted PCA

Compute the covariance matrix of the neighborhood:

```
c̄ = (1 / |N(i)|) · Σ_{j ∈ N(i)} v_j           (centroid)
C = (1 / |N(i)|) · Σ_{j ∈ N(i)} (v_j - c̄)(v_j - c̄)ᵀ
```

#### Weighted PCA (optional)

With Gaussian distance weighting:

```
w_j = exp(-||v_j - v_i||² / (2σ²))
w_j = w_j / Σ_k w_k
```

```
c̄_w = Σ_j w_j · v_j
C_w = Σ_j w_j · (v_j - c̄_w)(v_j - c̄_w)ᵀ
```

#### Eigen Decomposition

Solve the eigenvalue problem:

```
C · e_k = λ_k · e_k,    k = 0, 1, 2
```

where `λ₀ ≤ λ₁ ≤ λ₂` (sorted ascending).

- **Normal** = `e₀` (eigenvector with smallest eigenvalue — direction of least variance = surface normal)
- **Eigenvalue ordering**: `λ₀ ≤ λ₁ ≤ λ₂`

#### Normal Orientation

The normal is oriented outward (away from head centroid `c_head`):

```
if 〈e₀, v_i - c_head〉 < 0:
    n_i = -e₀
else:
    n_i = e₀
```

#### PCA Quality Metric

Quality measures how "planar" the local neighborhood is:

```
q_i = λ₀ / (λ₀ + λ₁ + λ₂)
```

- `q_i ≈ 0` → planar neighborhood (good — reliable normal)
- `q_i ≈ 1/3` → isotropic neighborhood (bad — unreliable normal)
- `q_i = 1.0` → fallback (insufficient neighbors)

### 2.2 ESE (Electrode Surface Equivalent) Generation

ESE is a virtual surface representing the expected positions of EEG electrode reference centers. Each ESE point is obtained by offsetting the scalp surface outward along the normal:

```
p_ese_i = p_scalp_i + d · n_i
```

where:
- `p_scalp_i` = vertex coordinates of the scalp mesh
- `n_i` = outward unit normal at vertex `i`
- `d` = offset distance in mm (`ESEConfig.offset_mm`, default 5.0 mm)

This produces a point cloud `P_ese = {p_ese₁, ..., p_ese_N}` that is `d` mm outside the scalp, following its curvature.

---

## Stage 3: Electrode Localization

### 3.1 Fiducial Points

Anatomical landmarks are defined in world coordinates:

| ID | Name | Location |
|----|------|----------|
| NAS | Nasion | Bridge of nose |
| LPA | Left Preauricular | Left of ear |
| RPA | Right Preauricular | Right of ear |
| INI | Inion | Back of skull (optional) |

Coordinates are stored as `F ∈ ℝ³`.

### 3.2 Distance Measurements

For each electrode `E_k`, measured distances to fiducials are provided:

```
m_k = {d(F_NAS, E_k), d(F_LPA, E_k), d(F_RPA, E_k)}
```

where `d(a, b) = ||a - b||₂` is Euclidean distance.

Minimum 2 fiducial distances required for localization.

### 3.3 Brute-Force Localization

For each electrode `E_k`, compute distances from **every** ESE point to all fiducials:

```
D_{i,j} = ||p_ese_i - F_j||₂,    i ∈ {1,...,N}, j ∈ {1,...,M}
```

where `N` = number of ESE points, `M` = number of fiducials.

**Cost function** — sum of squared residuals:

```
E(p_i) = Σ_{j=1}^{M} (D_{i,j} - m_{k,j})²
```

**Optimal position**:

```
p* = p_ese_{i*},    where i* = argmin_i E(p_i)
```

The electrode is placed at the ESE point with minimum cost.

### 3.4 Confidence Score

```
confidence = 1 / (1 + E(p*))
```

- `confidence → 1` when residual → 0 (perfect match)
- `confidence → 0` when residual → ∞ (poor match)

### 3.5 Residual Error

```
residual_k = E(p*_k) = Σ_j (D_{i*,j} - m_{k,j})²
```

- `residual < max_residual_threshold` (default 10 mm) → OK
- `residual > threshold` → electrode is **flagged** for review

### 3.6 Aggregated Metrics

```
mean_residual = (1/K) · Σ_{k=1}^{K} residual_k
max_residual = max_k residual_k
```

---

## Quality Control

### Stage 1 QC

| Check | Condition |
|-------|-----------|
| Affine valid | `∀ A_{ij} : not NaN ∧ not Inf` |
| Spatial metadata | `∀ i ∈ {0,1,2} : |A_{ii}| > 0` |
| Mesh non-empty | `|V| > 0`, `|F| > 0` |
| Mesh extent | `0 < extent_i < 1000 mm` for all axes |
| Face indices valid | `∀ f ∈ F : 0 ≤ f < |V|` |
| Fiducials ≥ 3 | `|fiducials| ≥ 3` |
| Fiducial distance | `min_dist(fid, surface) < 30 mm` |
| ESE offset positive | `d > 0` |
| Segmentation non-empty | `Σ M > 0` |

### Stage 2 QC

| Check | Condition |
|-------|-----------|
| ESE non-empty | `|P_ese| > 0` |
| Offset consistency | `std(d) < 0.1 · mean(d)` |
| Normals outward | `∀ i : 〈n_i, p_ese_i - c_head〉 ≥ 0` |
| Normal quality | `median(q) < 0.5` |
| Few outliers | `count(q > 0.8) < 0.05 · N` |

### Stage 3 QC

| Check | Condition |
|-------|-----------|
| Electrodes localized | `|electrodes| > 0` |
| Mean residual | `mean_residual < 20 mm` |
| Max residual | `max_residual < 50 mm` |
| Few flagged | `|flagged| < 0.2 · |electrodes|` |
| Min confidence | `min(confidence) > 0.01` |

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| `v_i` | Vertex coordinates (ℝ³) |
| `n_i` | Unit normal vector at vertex i |
| `p_ese_i` | ESE point coordinates |
| `F_j` | Fiducial point coordinates |
| `d(a, b)` | Euclidean distance `‖a - b‖₂` |
| `λ_k` | k-th eigenvalue of local covariance |
| `e_k` | k-th eigenvector of local covariance |
| `C` | Covariance matrix (3×3) |
| `A` | MRI affine transform (4×4) |
| `s` | Voxel spacing (ℝ³) |
| `σ` | Gaussian smoothing sigma |
| `d` | ESE offset distance (mm) |
| `λ` | Laplacian smoothing weight |
| `q` | PCA quality metric ∈ [0, 1] |
