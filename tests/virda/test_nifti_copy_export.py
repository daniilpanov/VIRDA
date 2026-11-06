import logging
from pathlib import Path

import nibabel as nib
import numpy as np

from virda.io.exporters.nifti_copy import export_nifti_copy


def _write_source(path: Path) -> None:
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4)), str(path))


class TestExportNiftiCopy:
    def test_copies_source_byte_for_byte(self, tmp_path: Path) -> None:
        source = tmp_path / "mri.nii.gz"
        _write_source(source)

        target = export_nifti_copy(tmp_path / "input" / "mri.nii.gz", source)

        assert target is not None
        assert target.read_bytes() == source.read_bytes()

    def test_replaces_previous_copy(self, tmp_path: Path) -> None:
        source = tmp_path / "mri.nii.gz"
        _write_source(source)
        stale = tmp_path / "input" / "mri.nii.gz"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"stale")

        target = export_nifti_copy(stale, source)

        assert target is not None
        assert target.read_bytes() == source.read_bytes()

    def test_returns_existing_target_when_it_is_the_source(self, tmp_path: Path) -> None:
        source = tmp_path / "input" / "mri.nii.gz"
        source.parent.mkdir(parents=True)
        _write_source(source)

        target = export_nifti_copy(source, source)

        assert target == source
        assert target.is_file()

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        source = tmp_path / "mri.nii.gz"
        _write_source(source)

        target = export_nifti_copy(tmp_path / "deep" / "input" / "mri.nii.gz", source)

        assert target is not None
        assert target.is_file()

    def test_reports_copy_failure_as_none(self, tmp_path: Path, caplog) -> None:
        missing = tmp_path / "missing.nii.gz"

        target = export_nifti_copy(
            tmp_path / "input" / "mri.nii.gz",
            missing,
            logger=logging.getLogger("virda.io.exporters.nifti_copy"),
        )

        assert target is None
        assert any(
            "Failed to copy source NIfTI" in record.getMessage() for record in caplog.records
        )
