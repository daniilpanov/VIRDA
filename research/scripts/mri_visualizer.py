# Script for reading and visualizing NIfTI files

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def load_nifti(filepath):
    """Load NIfTI file and extract data."""
    img = nib.load(filepath)
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    return data, affine, header, img


def print_info(header, data):
    """Print file information."""
    print("=" * 50)
    print("NIfTI File Information:")
    print("=" * 50)
    print(f"Data shape       : {data.shape}")
    print(f"Data type (dtype): {data.dtype}")
    print(f"Voxel size (mm)  : {header.get_zooms()}")
    print(f"Description      : {header.get('descrip')}")
    print(f"Min value        : {data.min():.4f}")
    print(f"Max value        : {data.max():.4f}")
    print(f"Mean value       : {data.mean():.4f}")
    print("=" * 50)


def plot_three_views(data, cmap="gray", title_prefix=""):
    """
    Display three orthogonal slices (axial, coronal, sagittal)
    through the volume center
    """
    # Find the volume center
    cx, cy, cz = [s // 2 for s in data.shape[:3]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Axial slice (XY plane)
    ax_slice = data[:, :, cz]
    axes[0].imshow(np.rot90(ax_slice), cmap=cmap, aspect="auto")
    axes[0].set_title(f"{title_prefix}Axial (z={cz})")
    axes[0].axis("off")

    # Coronal slice (XZ plane)
    cor_slice = data[:, cy, :]
    axes[1].imshow(np.rot90(cor_slice), cmap=cmap, aspect="auto")
    axes[1].set_title(f"{title_prefix}Coronal (y={cy})")
    axes[1].axis("off")

    # Sagittal slice (YZ plane)
    sag_slice = data[cx, :, :]
    axes[2].imshow(np.rot90(sag_slice), cmap=cmap, aspect="auto")
    axes[2].set_title(f"{title_prefix}Sagittal (x={cx})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def plot_slice_grid(data, axis="axial", num_slices=9, cmap="gray"):
    """
    Display a grid of slices along the selected axis.

    Parameters:
        data      : 3D numpy array
        axis      : 'axial' (z), 'coronal' (y), or 'sagittal' (x)
        num_slices: number of slices to display
        cmap      : matplotlib colormap
    """
    axis_map = {"axial": 2, "coronal": 1, "sagittal": 0}
    ax_idx = axis_map.get(axis, 2)
    num_total = data.shape[ax_idx]

    # Evenly distributed slice indices
    indices = np.linspace(0, num_total - 1, num_slices, dtype=int)

    cols = int(np.ceil(np.sqrt(num_slices)))
    rows = int(np.ceil(num_slices / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle(f"{axis.capitalize()} slices ({num_slices} of {num_total})", fontsize=14)

    for i, idx in enumerate(indices):
        r, c = divmod(i, cols)
        ax = axes[r, c] if rows > 1 else axes[c]

        if ax_idx == 2:
            slc = data[:, :, idx]
        elif ax_idx == 1:
            slc = data[:, idx, :]
        else:
            slc = data[idx, :, :]

        ax.imshow(np.rot90(slc), cmap=cmap, aspect="auto")
        ax.set_title(f"Slice {idx}", fontsize=9)
        ax.axis("off")

    # Hide empty subplots
    for j in range(num_slices, rows * cols):
        r, c = divmod(j, cols)
        ax = axes[r, c] if rows > 1 else axes[c]
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_histogram(data):
    """Histogram of voxel intensities."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(data.ravel(), bins=200, color="steelblue", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Count")
    ax.set_title("Voxel Intensity Distribution")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.show()


def plot_interactive_slider(data, axis="axial", cmap="gray"):
    """
    Interactive slice viewer using a slider (matplotlib widgets).
    """
    from matplotlib.widgets import Slider

    axis_map = {"axial": 2, "coronal": 1, "sagittal": 0}
    ax_idx = axis_map.get(axis, 2)
    num_slices = data.shape[ax_idx]

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.15)

    # Start slice - center
    mid = num_slices // 2

    def get_slice(idx):
        if ax_idx == 2:
            return data[:, :, idx]
        elif ax_idx == 1:
            return data[:, idx, :]
        else:
            return data[idx, :, :]

    im = ax.imshow(np.rot90(get_slice(mid)), cmap=cmap, aspect="auto")
    ax.axis("off")
    title = ax.set_title(f"{axis.capitalize()} — Slice {mid}/{num_slices - 1}")

    slider_ax = fig.add_axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(slider_ax, "Slice", 0, num_slices - 1, valinit=mid, valstep=1)

    def update(val):
        idx = int(slider.val)
        im.set_data(np.rot90(get_slice(idx)))
        title.set_text(f"{axis.capitalize()} — Slice {idx}/{num_slices - 1}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    import sys

    filepath = sys.argv[1] if len(sys.argv) > 1 else "example.nii"

    # Loading
    data, affine, header, img = load_nifti(filepath)

    # For fMRI
    if data.ndim == 4:
        print(f"Got 4D file {data.shape}, using only 3 first dimensions.")
        data = data[:, :, :, 0]

    # Info
    print_info(header, data)

    # 3 orthogonal slices
    plot_three_views(data, cmap="gray", title_prefix="")

    # Axial slices grid
    plot_slice_grid(data, axis="axial", num_slices=12, cmap="gray")

    # Voxels intensity histogram
    plot_histogram(data)

    # Interactive slider
    plot_interactive_slider(data, axis="axial", cmap="gray")
