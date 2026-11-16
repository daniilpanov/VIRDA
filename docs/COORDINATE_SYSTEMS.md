# Coordinate systems

A point on a head can be described from different starting points. VIRDA calls these
starting points **coordinate frames**. Every number in the program is a length in
**millimetres**, and the program always remembers which frame a number belongs to,
so a coordinate is never ambiguous.

The main numbers are stored in the **world** frame: the scanner's own coordinates.
Choosing another frame changes only how numbers are shown and exported, never the
stored values.

## The frames you can choose

| Frame | The numbers are measured from | When it is useful |
|---|---|---|
| **World** (default) | the scanner | comparing with the scanner data; the safest choice |
| **Head** | the centre of the head | comparing heads that sat differently in the scanner |
| **Fiducials** | the nasion (NAS) | comparing anatomically, independent of the head's size |

**World** is the frame the scanner's images are in. The scalp mesh, the ESE mesh,
the electrodes and the fiducials are stored in it, and it is the frame the exports
offer by default.

**Head** is the world translated so that the zero sits at the head's centre of mass
(the area-weighted centre of the scalp mesh). The directions stay the scanner's.
Use it when the head sat differently in the scanner: the same head then looks the
same in this frame.

**Fiducials** is the anatomical frame built from the three fiducials: NAS is the
origin and the directions follow the ears and the nasion (see below). Use it when
the head's size differs: the same anatomy lands on the same numbers.

## How a point moves between the frames

A frame is defined by its **origin** — the new zero point, in world millimetres —
and its **axes**, the three directions the numbers grow along. A world point `p`
converts into a frame as

```
p_frame = Rᵀ · (p − origin)
```

where `R` is the matrix whose columns are the frame's three axes. The way back is
`p = origin + R · p_frame`.

The **fiducials** axes are built from the three fiducials:

- the origin is **NAS**;
- the right axis is `normalize(RPA − LPA)`;
- the anterior axis is `normalize(NAS − mid(LPA, RPA))`, made perpendicular to the
  right axis;
- the up axis is `right × anterior`.

The fiducials themselves are never expressed in this frame: they *are* its
definition.

## When a frame is greyed out

A frame is unavailable while the project cannot provide it:

| Frame | Needs |
|---|---|
| **Head** | the scalp mesh (run the mesh steps first) |
| **Fiducials** | all three fiducials (NAS, LPA, RPA) |

The **world** frame is always available.

## The two surfaces: scalp and ESE

The pipeline works with two surfaces, and they are not the same thing:

| Surface | Where it comes from | What lives on it |
|---|---|---|
| Scalp | the head segmentation | the electrode **skin contact** points |
| ESE (electrode-skin-entrance) | the scalp surface, offset outward by the ESE offset | the electrode **body centre** points |

A localized electrode therefore carries two positions: the contact point on the
skin and the body centre on the ESE surface. The **ESE offset** is the distance
between the skin and the electrode body; when it is calibrated, the whole ESE
surface is shifted along the scalp normals to match the measured distances.

## The volume and the meshes

- **NIfTI (the volume)** keeps the scanner's own grid: the picture is a
  three-dimensional table of values, and the file also carries the mapping from
  that grid to the scanner's millimetres. The program reads the mapping and shows
  the volume, the meshes and the points in one piece; the numbers you enter and
  read in the tables still belong to the frame you chose.
- **PLY (the meshes)** stores the vertices in **world** millimetres — the scanner's
  coordinates. The scalp mesh, the postprocessed mesh and the ESE mesh are all in
  this frame, and so are the localized electrode points.

## About the files

- **`coordsystem.json` from your scanner or MNE** — VIRDA reads the digitized
  fiducials from it. The units are converted to millimetres, and the coordinates
  are understood as the scanner surface-RAS, which is the world frame. The file
  may describe the layout in metres (the BIDS default) or in millimetres; any other
  unit is rejected.
- **The project's `fiducials.json`** — written by VIRDA, always in millimetres.
  It may also store the fiducials in the volume's voxel indices; such a file needs
  the NIfTI volume to be interpreted.

## If something looks wrong

- The numbers are shifted → check the frame selector: the numbers are shown in the
  selected frame, while the stored values stay in world.
- A frame is greyed out → run the steps it needs: the scalp mesh for **Head**, the
  three fiducials for **Fiducials**.
- The export carries different numbers than the table → the export dialog has its
  own frame selector; both default to **World**.
