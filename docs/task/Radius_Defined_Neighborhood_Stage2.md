# Stage 2 – Formal Definition of the Radius-Defined Neighborhood

**Purpose**
This section formally defines the local neighborhood used for principal component analysis (PCA) to estimate the local scalp surface normal at each mesh vertex.

## 1. Scalp Surface Mesh

Let S = {p1, p2, ..., pN} denote the set of vertices of the segmented scalp surface mesh, where each vertex pi=(xi, yi, zi)^T is expressed in the MRI (head) coordinate system.

## 2. Radius-Defined Neighborhood

For an arbitrary mesh vertex pi, define its local neighborhood N(pi) as the subset of mesh vertices lying within a fixed Euclidean distance r from pi.

N(pi) = { pj ∈ S : ||pj − pi|| ≤ r }

where ||·|| denotes the Euclidean norm and r is the predefined neighborhood radius (typically several millimetres).

## 3. Neighbor and Non-neighbor Points

A mesh vertex pj is called a neighbor of pi if ||pj − pi|| ≤ r.
A mesh vertex pj is called a non-neighbor of pi if ||pj − pi|| > r.

## 4. Purpose of the Neighborhood

Only the vertices belonging to N(pi) are used to construct the local data matrix for PCA. Vertices outside the neighborhood are excluded because they represent anatomically distant regions of the scalp and could bias estimation of the local tangent plane and surface normal.

## 5. Practical Considerations

The neighborhood radius should be sufficiently large to suppress MRI segmentation noise, yet sufficiently small to preserve genuine local scalp curvature. Typical candidate radii for evaluation are 5 mm, 10 mm and 15 mm. The optimal value may be selected empirically.

## 6. Output

For every scalp mesh vertex, the radius-defined neighborhood is used to estimate the local tangent plane by PCA. The resulting outward normal is then used to calculate the corresponding External Scalp Envelope (ESE) point in the original MRI head coordinate system.
