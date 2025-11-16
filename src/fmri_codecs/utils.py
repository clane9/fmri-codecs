import urllib.request
from pathlib import Path

import numpy as np
from nibabel.cifti2 import BrainModelAxis, Cifti2Image


def fetch_schaefer(num_rois: int) -> Path:
    base_url = (
        "https://github.com/ThomasYeoLab/CBIG/raw/refs/heads/master/"
        "stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/"
        "Parcellations/HCP/fslr32k/cifti/"
    )
    filename = f"Schaefer2018_{num_rois}Parcels_17Networks_order.dscalar.nii"
    url = f"{base_url}/{filename}"

    cache_dir = Path.home() / ".cache" / "schaefer"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached_file = cache_dir / filename
    if not cached_file.exists():
        urllib.request.urlretrieve(url, cached_file)
        assert cached_file.exists(), f"Download failed: {url}"
    return cached_file


def get_cifti_surf_data(cifti: Cifti2Image) -> np.ndarray:
    lh_data = get_cifti_struct_data(cifti, "CIFTI_STRUCTURE_CORTEX_LEFT")
    rh_data = get_cifti_struct_data(cifti, "CIFTI_STRUCTURE_CORTEX_RIGHT")
    data = np.concatenate([lh_data, rh_data], axis=0)
    return data


def get_cifti_struct_data(cifti: Cifti2Image, struct: str) -> np.ndarray:
    """Get cifti scalar/series data for a given brain structure."""
    axis = get_brain_model_axis(cifti)
    data = cifti.get_fdata().T
    for name, indices, model in axis.iter_structures():
        if name == struct:
            num_verts = model.vertex.max() + 1
            struct_data = np.zeros((num_verts,) + data.shape[1:], dtype=data.dtype)
            struct_data[model.vertex] = data[indices]
            return struct_data
    raise ValueError(f"Invalid cifti struct {struct}")


def get_brain_model_axis(cifti: Cifti2Image) -> BrainModelAxis:
    for ii in range(cifti.ndim):
        axis = cifti.header.get_axis(ii)
        if isinstance(axis, BrainModelAxis):
            return axis
    raise ValueError("No brain model axis found in cifti")
