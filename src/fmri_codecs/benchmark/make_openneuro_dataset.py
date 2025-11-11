import json
from pathlib import Path

import datasets as hfds
import numpy as np
import nibabel as nib
from nibabel.cifti2 import BrainModelAxis, Cifti2Image
from sklearn.preprocessing import scale

# Number of total fslr vertices across cortex
NUM_VERTICES = 64984

EPS = 1e-6
NUM_PROC = 16

ROOT = Path(__file__).parents[3]


def main():
    file_splits_path = ROOT / "config/openneuro_file_splits.json"
    with file_splits_path.open() as f:
        file_splits = json.load(f)

    features = hfds.Features(
        {
            "dataset": hfds.Value("string"),
            "sub": hfds.Value("string"),
            "ses": hfds.Value("string"),
            "task": hfds.Value("string"),
            "run": hfds.Value("int32"),
            "acq": hfds.Value("string"),
            "n_frames": hfds.Value("int32"),
            "bold": hfds.Array2D(shape=(None, NUM_VERTICES), dtype="float16"),
        }
    )

    dataset_dict = {}
    for split, paths in file_splits.items():
        dataset_dict[split] = hfds.Dataset.from_generator(
            generate_samples,
            features=features,
            gen_kwargs={"root": ROOT / "datasets/openneuro", "paths": paths},
            num_proc=NUM_PROC,
            split=hfds.NamedSplit(split),
        )

    dataset = hfds.DatasetDict(dataset_dict)
    dataset.push_to_hub("clane9/openneuro-fslr64k", max_shard_size="600MB", num_proc=NUM_PROC)


def generate_samples(root: Path, paths: list[str]):
    for path in paths:
        path = root / path
        meta = parse_metadata(path)

        img = nib.load(path)
        series = get_cifti_surf_data(img)
        series = np.ascontiguousarray(series.T)

        T, D = series.shape
        assert D == NUM_VERTICES

        valid_mask = np.std(series, axis=0) > EPS
        series = scale(series)
        series = series * valid_mask
        series = series.astype(np.float16)

        n_frames = len(series)

        sample = {**meta, "n_frames": n_frames, "bold": series}
        yield sample


def parse_metadata(path: Path):
    dataset = next(part for part in path.parts if part.startswith("ds"))
    dataset, _ = dataset.split("-")
    meta = dict(item.split("-") for item in path.stem.split("_") if "-" in item)
    meta = {k: meta.get(k) for k in ["sub", "ses", "task", "run", "acq"]}
    if meta["run"] is not None:
        meta["run"] = int(meta["run"])
    meta = {"dataset": dataset, **meta}
    return meta


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


if __name__ == "__main__":
    main()
