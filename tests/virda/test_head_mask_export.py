import nibabel as nib
import numpy as np
import pytest

from virda.io.exporters.segmentation import export_head_mask
from virda.models.mri_volume import MRIVolume
from virda.models.segmentation_mask import SegmentationMask


@pytest.fixture()
def volume() -> MRIVolume:
    affine = np.diag([1.5, 1.5, 2.0, 1.0])
    return MRIVolume(
        data=np.zeros((4, 4, 4), dtype=np.float32),
        affine=affine,
        spacing=(1.5, 1.5, 2.0),
        orientation=("R", "A", "S"),
    )


class TestExportHeadMask:
    def test_writes_uint8_nifti_with_volume_affine(self, tmp_path, volume) -> None:
        mask = np.random.randint(0, 2, size=(4, 4, 4)).astype(np.float32)
        volume = MRIVolume(
            data=mask,
            affine=volume.affine,
            spacing=volume.spacing,
            orientation=volume.orientation,
        )

        path = export_head_mask(tmp_path / "head_mask.nii.gz", SegmentationMask(mask), volume)

        image = nib.load(path)
        assert isinstance(image, nib.Nifti1Image)
        assert image.header.get_data_dtype() == np.dtype(np.uint8)
        np.testing.assert_array_equal(image.dataobj.get_unscaled(), mask.astype(np.uint8))
        np.testing.assert_array_almost_equal(image.affine, volume.affine)

    def test_creates_parent_directory(self, tmp_path, volume) -> None:
        path = export_head_mask(
            tmp_path / "deep" / "segmentation" / "head_mask.nii.gz",
            SegmentationMask(np.zeros((2, 2, 2), dtype=np.float32)),
            volume,
        )

        assert path.is_file()
