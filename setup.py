from setuptools import setup, find_packages

setup(
    name="virda",
    version="0.2.0",
    description="VIRDA Electrode Localization from MRI",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-image",
        "trimesh",
    ],
    extras_require={
        "mri": ["nibabel", "pydicom", "mne"],
        "viz": ["pyvista", "matplotlib"],
        "all": ["nibabel", "pydicom", "mne", "pyvista", "matplotlib"],
        "dev": ["pytest"],
    },
)
