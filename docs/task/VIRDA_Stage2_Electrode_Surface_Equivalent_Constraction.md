For each scalp-mesh point, PCA is applied not to the whole head, but to a **small local neighborhood** around that point.

Let the mesh point of interest be

$$p_0 = \begin{pmatrix} x_0 \\ y_0 \\ z_0 \end{pmatrix}.$$

Choose **neighboring surface points:**

$$N$$

$$p_i = \begin{pmatrix} x_i \\ y_i \\ z_i \end{pmatrix}, \quad i = 1, \dots, N.$$

These can be the nearest mesh vertices or all vertices within a selected radius.

**1. Construct the raw data matrix**

Place the coordinates of the neighboring points in rows:

$$P = \begin{pmatrix} x_1 & y_1 & z_1 \\ x_2 & y_2 & z_2 \\ \vdots & \vdots & \vdots \\ x_N & y_N & z_N \end{pmatrix}.$$

Thus:
- each row represents one neighboring scalp point;
- the three columns are the $x$, $y$, and $z$ coordinates;
- the matrix has dimensions $N \times 3$.

You may include $p_0$ itself in the neighborhood, although this is not essential.

**2. Center the coordinates**

PCA must be performed on coordinates relative to the local neighborhood center, not on the original MRI coordinates.

Calculate the centroid:

$$\bar{p} = \frac{1}{N} \sum_{i} p_i = \begin{pmatrix} \bar{x} \\ \bar{y} \\ \bar{z} \end{pmatrix}.$$

Then subtract the centroid from every row:

$$X = \begin{pmatrix} x_1 - \bar{x} & y_1 - \bar{y} & z_1 - \bar{z} \\ x_2 - \bar{x} & y_2 - \bar{y} & z_2 - \bar{z} \\ \vdots & \vdots & \vdots \\ x_N - \bar{x} & y_N - \bar{y} & z_N - \bar{z} \end{pmatrix}.$$

This centered $N \times 3$ matrix $X$ is the actual PCA data matrix.

**3. Construct the covariance matrix**

The local covariance matrix is

$$C = \frac{1}{N-1} X^T X.$$

Because $X$ has three coordinate columns, $C$ is a $3 \times 3$ matrix:

$$C = \begin{pmatrix} C_{xx} & C_{xy} & C_{xz} \\ C_{yx} & C_{yy} & C_{yz} \\ C_{zx} & C_{zy} & C_{zz} \end{pmatrix}.$$

This matrix describes how the neighboring surface points vary in three-dimensional space.

**4. Calculate eigenvalues and eigenvectors**

Solve

$$C v_j = \lambda_j v_j.$$

Sort the eigenvalues:

$$\lambda_1 \geq \lambda_2 \geq \lambda_3.$$

The corresponding eigenvectors are

$$v_1, \quad v_2, \quad v_3.$$

Their geometric meanings are:
- $v_1$: direction of greatest variation along the surface;
- $v_2$: second direction along the surface;
- $v_3$: direction of least variation.

Because the points lie approximately on a local surface patch, they vary substantially in two tangential directions but very little perpendicular to the surface. Therefore,

$$n = v_3$$

is the estimated local surface normal.

In other words, the eigenvector corresponding to the **smallest eigenvalue** is the normal.

Conceptually, PCA fits the plane that best approximates the local points. The normal is perpendicular to that plane.

**5. Determine whether the normal points outward**

PCA provides the normal direction only up to sign:

$$n \text{ and } -n$$

are mathematically equivalent. You must therefore orient it outward.

A simple approach is to define an approximate interior point of the head, such as the centroid of the full scalp surface:

$$c_{\text{head}}.$$

Then calculate

$$r = p_0 - c_{\text{head}}.$$

If

$$n \cdot r < 0,$$

reverse the normal:

$$n \leftarrow -n.$$

After this operation, $n$ points approximately away from the head.

For a generally convex scalp surface, this works well. Around strongly concave regions, a mesh-based orientation method may be more reliable.

**6. Generate the ESE point**

Once the outward unit normal is known, calculate

$$p_{\text{ESE}} = p_0 + d\, n,$$

where $d$ is the distance from the MRI scalp surface to the chosen electrode external reference surface.

For example, if

$$p_0 = \begin{pmatrix} 20 \\ 40 \\ 80 \end{pmatrix} \text{ mm}, \quad n = \begin{pmatrix} 0.2 \\ 0.3 \\ 0.9327 \end{pmatrix},$$

and the offset is $d = 5$ mm, then

$$p_{\text{ESE}} = \begin{pmatrix} 20 \\ 40 \\ 80 \end{pmatrix} + 5 \begin{pmatrix} 0.2 \\ 0.3 \\ 0.9327 \end{pmatrix} = \begin{pmatrix} 21 \\ 41.5 \\ 84.6635 \end{pmatrix} \text{ mm}.$$

**A small numerical illustration**

Suppose the neighborhood contains six points:

$$P = \begin{pmatrix} -1 & -1 & 10.1 \\ 0 & -1 & 10.0 \\ 1 & -1 & 9.9 \\ -1 & 1 & 10.0 \\ 0 & 1 & 10.1 \\ 1 & 1 & 10.0 \end{pmatrix}.$$

The points vary strongly in $x$ and $y$, but only slightly in $z$. After centering and PCA:
- the two largest eigenvectors will lie approximately in the $xy$-plane;
- the smallest eigenvector will be approximately

$$v_3 \approx \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}.$$

Therefore, the local surface normal is approximately vertical.

**Choosing the neighborhood**

This is important for VIRDA because the scalp mesh can contain MRI segmentation noise.

A very small neighborhood gives a highly local estimate but may produce noisy normals. A very large neighborhood gives smooth normals but may overlook genuine local curvature.

Two practical methods are:

**Fixed number of neighbors**

For each vertex, take the $k$ nearest mesh vertices, for example:

$$k = 20, \quad 30, \quad \text{or} \quad 50.$$

This is convenient when mesh density is relatively uniform.

**Fixed physical radius**

Take all points within a radius such as:

$$r = 5 \text{ -- } 10 \text{ mm}.$$

This is often preferable because it corresponds to a real anatomical scale. For VIRDA, it may be useful to test several scales, including a radius related to the electrode dimensions.

I would initially test perhaps $5$, $8$, and $10$ mm neighborhoods and visually compare the resulting normal fields.

**Weighted PCA** &nbsp;&nbsp;**(OPTIONAL)**

A useful refinement is to give nearby points more influence than distant points. Define a weight, for example,

$$w_i = \exp\frac{\|p_i - p_0\|^2}{2\sigma^2}.$$

Then use the weighted centroid

$$p_w = \frac{\sum_i w_i p_i}{\sum_i w_i},$$

and weighted covariance

$$C_w = \frac{1}{\sum_i w_i} \sum_i w_i (p_i - p_w)(p_i - p_w)^T.$$

The eigenvector of $C_w$ with the smallest eigenvalue is again the normal. Weighted PCA can reduce abrupt changes caused by the boundary of the selected neighborhood.

**One additional quality measure**

The eigenvalues can tell you how reliable the estimated normal is.

For a good locally planar surface patch,

$$\lambda_3 \ll \lambda_1, \quad \lambda_2.$$

A useful measure is

$$q = \frac{\lambda_3}{\lambda_1 + \lambda_2 + \lambda_3}.$$

A small $q$ indicates that the points lie close to a plane and the normal estimate is reliable. A larger $q$ may indicate:
- segmentation noise;
- a sharp anatomical feature;
- an unsuitable neighborhood size;
- insufficient mesh density.

Thus, PCA provides not only a normal vector but also a local estimate of confidence.
