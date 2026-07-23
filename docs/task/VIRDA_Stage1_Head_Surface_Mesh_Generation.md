**VIRDA Electrode Localization**

**Stage 1: MRI Head Surface Mesh Generation and Introduction of the Electrode Surface Equivalent (ESE)**

*Detailed Programmer-Oriented Specification*

# 1. Purpose of Stage 1

The purpose of Stage 1 is to create a patient-specific three-dimensional representation of the external head surface from MRI data. This representation will serve as the anatomical foundation for all later VIRDA electrode-localization steps.

Stage 1 also introduces the Electrode Surface Equivalent (ESE). The ESE is a virtual external surface representing where a selected reference point on the VIRDA electrode is expected to lie after accounting for the physical distance between the scalp and the electrode external surface or center.

Stage 1 does not yet localize real electrodes. Its task is to prepare a clean, anatomically correct and geometrically usable head-surface model in the MRI coordinate system.

# 2. Main Objectives

- Load a patient MRI while preserving its spatial coordinate information.
- Segment the external head surface.
- Generate a triangular mesh of the scalp and relevant facial surface.
- Clean and preprocess the mesh without significantly changing the patient anatomy.
- Store anatomical fiducials in the same coordinate system as the mesh.
- Define the ESE concept and the electrode offset parameter.
- Prepare all data structures needed for Stage 2, where local surface normals and scalp-to-ESE point pairs will be calculated.

# 3. Scope and Boundaries

Included in Stage 1:
- MRI import.
- Coordinate-system preservation.
- External head segmentation.
- Surface extraction.
- Mesh cleaning and optional smoothing.
- Anatomical fiducial management.
- Definition of ESE parameters.
- Visualization and data export.

Not included in Stage 1:
- PCA calculation of surface normals.
- Generation of scalp-to-ESE point pairs.
- Localization of real electrodes from measured distances.
- EEG forward modeling or source localization.

# 4. Required Inputs

## 4.1 MRI data

The preferred input is a three-dimensional T1-weighted anatomical MRI. The program should ideally support DICOM series and NIfTI files. The input must include enough spatial metadata to convert voxel indices into physical coordinates measured in millimeters.
- MRI volume.
- Voxel dimensions.
- Image orientation.
- Voxel-to-world transformation matrix.
- Patient or scanner coordinate-system information when available.

## 4.2 Optional supporting inputs

- Pre-existing head or scalp segmentation.
- Pre-existing surface mesh.
- Manually supplied anatomical fiducial coordinates.
- A configuration file containing mesh-processing parameters.

# 5. Coordinate-System Requirements

All outputs from Stage 1 must remain linked to the original MRI coordinate system. This is critical because later stages will express ESE points and real electrode positions in the same patient-specific head coordinates.

The program should store both voxel coordinates and world coordinates when possible:
- Voxel coordinates: integer indices such as row, column and slice.
- World coordinates: physical x, y and z coordinates in millimeters.

The software must never assume that voxel axes correspond directly to anatomical left-right, anterior-posterior or inferior-superior directions. Orientation must be read from image metadata.

# 6. External Head Segmentation

## 6.1 Goal

Create a binary or labeled volume that separates the patient head from surrounding air and non-anatomical objects.

## 6.2 Desired anatomical coverage

- Entire scalp.
- Forehead.
- Nasion region.
- Temporal regions.
- Preauricular regions when visible.
- Posterior head and inion region.
- Sufficient facial surface to support anatomical registration.

## 6.3 Objects that should be excluded when possible

- Head coil components.
- Pillow and table.
- Blankets and clothing.
- External cables.
- Imaging artifacts disconnected from the head.

## 6.4 Segmentation approaches

The first version may use an established image-processing library or external segmentation tool rather than implementing a complete segmentation method from scratch. Possible approaches include thresholding, connected-component analysis, morphological operations, region growing or an existing medical-image segmentation package.

The programmer should design the module so that the segmentation method can later be replaced without changing the remainder of the pipeline.

# 7. Surface Mesh Generation

## 7.1 Mesh extraction

Generate a triangular surface mesh from the external head segmentation. A standard method such as marching cubes is acceptable.

## 7.2 Required mesh data

- Vertex list: one x, y, z coordinate for each point.
- Triangle list: three vertex identifiers for each triangle.
- Vertex-to-vertex or vertex-to-face neighborhood information.
- Optional preliminary face normals and vertex normals for visualization.
- Transformation information linking the mesh to MRI coordinates.

## 7.3 Suggested mesh object

| Field | Description |
| --- | --- |
| vertices | Array of N rows and 3 columns containing x, y, z coordinates. |
| faces | Array of M rows and 3 columns containing vertex indices. |
| adjacency | Neighborhood information for each vertex. |
| coordinate_system | Description of the MRI/world coordinate convention. |
| transform | Voxel-to-world or mesh-to-MRI transformation matrix. |
| metadata | Patient-independent processing settings and file provenance. |

# 8. Mesh Preprocessing

## 8.1 Cleaning operations

- Remove small disconnected components.
- Fill small holes when appropriate.
- Remove duplicate vertices and degenerate triangles.
- Correct inconsistent triangle orientation.
- Check for self-intersections or non-manifold areas when possible.

## 8.2 Smoothing

Controlled smoothing may be applied to reduce MRI segmentation noise. However, excessive smoothing can change the patient-specific head shape and shift anatomical fiducials. The amount of smoothing must therefore be configurable and recorded.

## 8.3 Mesh density

The mesh should be dense enough to support local surface-normal estimation in Stage 2. Optional mesh decimation may be used for speed, but the software should preserve a high-resolution version or document the reduction parameters.

# 9. Anatomical Fiducials

## 9.1 Purpose

Anatomical fiducials provide reference points that connect MRI anatomy with external measurements made on the patient. They will be especially important in Stage 3.

## 9.2 Suggested fiducials

- Nasion.
- Left preauricular point.
- Right preauricular point.
- Inion, when reliably visible or manually defined.
- Additional stable cranial or facial landmarks.

## 9.3 Fiducial data structure

| Field | Description |
| --- | --- |
| fiducial_id | Unique label such as NAS, LPA, RPA or INI. |
| name | Human-readable name. |
| coordinates | x, y, z coordinates in MRI millimeters. |
| definition_method | Manual, automatic or imported. |
| confidence | Optional quality or confidence value. |
| notes | Free-text comments. |

# 10. Introduction of the ESE Concept

## 10.1 Definition

The Electrode Surface Equivalent is a virtual surface outside the MRI-derived scalp. It represents the expected location of a selected electrode reference surface or electrode reference center after accounting for electrode height above the scalp.

In the first implementation, the ESE should be treated as a constant-distance offset from the scalp surface. The actual offset points will be calculated in Stage 2 after the local outward surface normals are known.

## 10.2 ESE reference definition

Before coding, the team must choose exactly what the ESE represents. Examples include:
- The center of the external upper surface of the electrode capsule.
- The geometric center of the electrode body.
- Another reproducible point that can be identified by photogrammetry or manual measurement.

The same definition must be used in the MRI model, external measurements and Stage 3 localization. A mismatch in this definition would create systematic localization error.

## 10.3 Electrode offset parameter

The ESE offset should be stored as a configurable value in millimeters. In the first version it may be constant for all points. Future versions may allow local variation caused by hair, compression, electrode type or cap deformation.

# 11. Suggested Software Modules

| Module | Responsibility | Primary output |
| --- | --- | --- |
| MRI Loader | Read DICOM or NIfTI and preserve spatial metadata. | MRI volume and coordinate transform. |
| Head Segmenter | Separate the external head from background. | Binary or labeled head volume. |
| Surface Extractor | Generate a triangular surface. | Raw scalp mesh. |
| Mesh Cleaner | Remove artifacts and repair mesh defects. | Clean scalp mesh. |
| Fiducial Manager | Create, edit and store anatomical landmarks. | Fiducial table. |
| ESE Configuration | Store the ESE reference definition and offset. | ESE settings. |
| 3D Viewer | Display MRI, mesh and fiducials. | Quality-control visualization. |
| Exporter | Save mesh and metadata in standard formats. | Reusable Stage 1 dataset. |

# 12. Step-by-Step Processing Workflow

1. Select and load the MRI dataset.
2. Read voxel spacing, orientation and voxel-to-world transformation.
3. Create or import the external head segmentation.
4. Keep the largest anatomically relevant connected component and remove obvious external objects.
5. Generate a triangular surface mesh.
6. Transform mesh vertices into MRI world coordinates.
7. Clean the mesh and apply only controlled smoothing.
8. Display the mesh together with MRI slices for visual inspection.
9. Mark or import anatomical fiducials.
10. Define and store the ESE reference point and offset distance.
11. Export the complete Stage 1 dataset for use by Stage 2.

# 13. Quality-Control Requirements

The Stage 1 output should not be accepted automatically. The program should provide both automatic checks and visual review.

## 13.1 Automatic checks

- MRI affine or spatial transformation is present and valid.
- Mesh contains vertices and triangles.
- Mesh coordinates are in millimeters.
- No large disconnected components remain.
- No major holes are present over the electrode-bearing scalp.
- Fiducial coordinates fall near the external head surface.
- ESE offset is positive and recorded.

## 13.2 Visual checks

- Overlay the mesh on axial, coronal and sagittal MRI slices.
- Confirm that the mesh follows the actual scalp.
- Confirm that the forehead, temporal areas and posterior scalp are represented correctly.
- Confirm that fiducials are anatomically plausible.
- Inspect for spikes, holes, bridges or accidental inclusion of the pillow or head coil.

# 14. Recommended File Outputs

- Scalp mesh: PLY, VTK, OBJ or STL.
- Vertex and face arrays: NumPy, CSV or JSON as appropriate.
- Fiducial table: CSV or JSON.
- Coordinate transforms and processing parameters: JSON.
- Optional segmentation mask: NIfTI.
- Optional quality-control screenshots.

A single project folder should contain all outputs together with a machine-readable metadata file so that Stage 2 can be reproduced exactly.

# 15. Suggested Stage 1 Project Folder

Example folder structure:

```
patient_project/
  input_mri/
  segmentation/
  mesh/
  fiducials/
  config/
  quality_control/
  logs/
```

# 16. Error Handling

- Report missing or inconsistent MRI spatial metadata.
- Do not silently continue if the coordinate transform is invalid.
- Warn if segmentation produces multiple large components.
- Warn if the mesh is empty, non-manifold or extremely sparse.
- Warn if a fiducial lies far from the head surface.
- Store processing errors and warnings in a log file.

# 17. Validation Strategy

## 17.1 Synthetic validation

Begin with a sphere or ellipsoid whose correct surface is known. Confirm that the generated mesh has the expected dimensions and coordinates.

## 17.2 Realistic head-mesh validation

Use a publicly available or manually prepared head mesh to test mesh cleaning, fiducial placement and export.

## 17.3 MRI validation

Use a patient MRI and compare the generated scalp mesh against the original images in all three anatomical planes.

## 17.4 Reproducibility

Running the same input with the same settings should produce the same mesh and fiducial coordinates.

# 18. Minimum Success Criteria

1. The program loads an MRI and preserves its physical coordinate system.
2. The external head surface is segmented without major non-anatomical objects.
3. A usable triangular scalp mesh is generated.
4. Mesh vertices are expressed in MRI world coordinates.
5. The mesh can be visualized over the MRI.
6. Anatomical fiducials can be entered, edited, saved and reloaded.
7. The ESE reference definition and offset distance can be configured.
8. All outputs can be exported and loaded by Stage 2.

# 19. Future Extensions

- Automatic anatomical fiducial detection.
- Hair-layer estimation.
- Separate facial and scalp regions.
- Multiple ESE definitions for different electrode types.
- Cap compression and deformation models.
- Direct import of meshes generated by established neuroimaging software.
- Integration with the later patient-specific external coordinate manifold concept.

# 20. Final Deliverable of Stage 1

The final deliverable is a validated, patient-specific MRI head-surface dataset containing:
- A clean scalp/head surface mesh.
- MRI spatial transformation information.
- Anatomical fiducial coordinates.
- Mesh neighborhood/connectivity information.
- ESE reference definition and offset settings.
- Quality-control results and processing metadata.

This dataset becomes the direct input to Stage 2, where PCA will be used to estimate local outward normals and construct explicit scalp-point to ESE-point pairs.
