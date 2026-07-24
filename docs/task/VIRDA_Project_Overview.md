# VIRDA Electrode Localization Project
## Programmer-Oriented Overview

This document describes the project in software-engineering terms. It intentionally avoids advanced mathematical notation and focuses on the software modules, data structures, algorithms, inputs and outputs.

## Overall Goal

Develop software that localizes VIRDA electrodes in the patient's MRI coordinate system. The program should build a virtual scalp model from MRI, generate a virtual Electrode Surface Equivalent (ESE), and estimate the positions of real electrodes from external measurements.

## Stage 1 - MRI Head Surface and ESE

Tasks:
- Load MRI volume.
- Segment the external head surface.
- Generate a triangular scalp mesh.
- Store vertices, triangles and neighborhood information.
- Allow manual marking of anatomical fiducials (nasion, left/right preauricular points, etc.).
- Define an electrode offset distance (for example 5 mm).
- Prepare data structures for normal-vector calculation.

Output:
- Scalp mesh.
- Fiducial coordinates.
- Mesh connectivity.
- Configurable electrode offset parameter.

## Stage 2 - Construct the Electrode Surface Equivalent (ESE)

For every scalp mesh vertex:
1. Find neighboring mesh points.
2. Create a local coordinate matrix containing the x,y,z coordinates of the neighbors.
3. Center the coordinates by subtracting the local average.
4. Run PCA on the centered coordinates.
5. The eigenvector with the smallest variance is the local surface normal.
6. Orient the normal outward.
7. Move the point outward by the electrode offset distance.
8. Store the pair: ScalpPoint <-> ESEPoint.

Recommended data structure for each point:
- Point ID
- Scalp coordinates
- ESE coordinates
- Normal vector
- PCA quality value

## Stage 3 - Localization of Real Electrodes

Input:
- Coordinates of MRI fiducials.
- Measured distances from each electrode center to several fiducials (obtained by photogrammetry or manually).

Algorithm:
1. For each electrode, search candidate positions on the ESE.
2. Calculate predicted distances from the candidate to each fiducial.
3. Compare predicted distances with measured distances.
4. Choose the candidate with the smallest total error.
5. Report localization error.
6. Convert the ESE position back to the corresponding scalp point.

## Suggested Software Modules

- MRI Loader
- Head Segmentation
- Mesh Generator
- PCA Normal Estimator
- ESE Generator
- Fiducial Manager
- Measurement Import
- Electrode Localizer
- 3D Visualization
- Export Module

## Recommended Development Order

1. Prototype using a sphere.
2. Prototype using a realistic head mesh.
3. Prototype using a patient MRI.
4. Finally test with real VIRDA measurements.

## Success Criteria

- Generate a scalp mesh from MRI.
- Construct the ESE.
- Import fiducial measurements.
- Estimate electrode locations.
- Visualize the result in 3D.
- Export coordinates for future source localization.
