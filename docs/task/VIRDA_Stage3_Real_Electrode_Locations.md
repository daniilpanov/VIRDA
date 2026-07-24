# VIRDA Electrode Localization
## Stage 3: Localization of Real Electrodes from External Measurements

### Purpose

The objective of Stage 3 is to determine the three-dimensional location of each real VIRDA electrode in the patient's MRI head coordinate system. The algorithm uses the previously constructed Electrode Surface Equivalent (ESE) together with externally measured distances from electrode centers to anatomical fiducials.

### Inputs

- Patient-specific scalp mesh and ESE generated in Stages 1 and 2.
- Coordinates of anatomical fiducials (e.g. nasion, left/right preauricular points, optional inion).
- Measured distances from each electrode center to the fiducials (photogrammetry or manual calipers).
- Optional measured distances between neighboring electrodes.

### Concept

Each real electrode is assumed to lie on or very close to the ESE. The software searches the ESE for the point whose distances to the anatomical fiducials best match the measured distances. This converts external measurements into MRI coordinates.

### Algorithm

1. Load the ESE point cloud or triangular mesh.
2. Load fiducial coordinates.
3. For one electrode, read the measured distances to each fiducial.
4. For every candidate point on the ESE, calculate the predicted Euclidean distance to each fiducial.
5. Calculate the total localization error as the sum (or weighted sum) of the squared differences between measured and predicted distances.
6. Select the candidate with the smallest total error.
7. Store the selected ESE coordinates.
8. Recover the corresponding scalp point using the point-pair table from Stage 2.
9. Repeat for all electrodes.

### Possible Optimization

The first implementation may simply test every ESE vertex (brute-force search). Later versions may use optimization methods, KD-trees, or continuous optimization on the triangular mesh to improve speed and accuracy.

### Handling Measurement Error

- Allow different confidence weights for different fiducials.
- Ignore missing measurements.
- Calculate the residual error for every electrode.
- Flag electrodes whose residual error exceeds a user-defined threshold.

### Suggested Data Structure

| Field | Description |
| --- | --- |
| Electrode ID | Unique identifier |
| Measured distances | Distances to fiducials |
| Estimated ESE coordinates | Best-fit ESE point |
| Corresponding scalp coordinates | Mapped scalp point |
| Residual error | Localization error |
| Confidence | Quality indicator |

### Visualization

- MRI scalp mesh
- ESE surface
- Anatomical fiducials
- Estimated electrode centers
- Lines from electrodes to fiducials
- Localization error map

### Outputs

- MRI coordinates of every localized electrode.
- Associated scalp coordinates.
- Residual localization errors.
- CSV/JSON export of electrode coordinates.
- 3D visualization for quality control.

### Future Extensions

After validation of the point-to-point approach, future versions may incorporate finite electrode footprints, patch-to-patch registration, cap deformation, probabilistic localization, and integration with EEG forward models.
