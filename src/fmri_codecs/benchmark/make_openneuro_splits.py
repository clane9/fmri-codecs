import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 4616
TOTAL_NUM_SERIES = 9373

FRAME_SIZE_BYTES = 91282 * 2
SIZE_RANGE = 100 * FRAME_SIZE_BYTES, 300 * FRAME_SIZE_BYTES

SUBS_PER_DATASET = 20
TRAIN_FRACTION = 0.5

ROOT = Path(__file__).parents[3]


def main():
    rng = np.random.default_rng(SEED)

    root = ROOT / "datasets/openneuro"
    series_paths = sorted(root.rglob("*_space-fsLR_den-91k_bold.dtseries.nii"))
    print(f"num series: {len(series_paths)}")
    assert len(series_paths) == TOTAL_NUM_SERIES

    records = []
    for path in series_paths:
        meta = parse_metadata(path)
        records.append({**meta, "size": path.stat().st_size, "path": str(path)})
    df = pd.DataFrame.from_records(records)

    size_min, size_max = SIZE_RANGE
    size_mask = (df["size"] >= size_min) & (df["size"] <= size_max)
    df = df.loc[size_mask]
    print(f"num filtered series: {len(df)}")

    ds_counts = df.groupby("dataset").agg({"sub": "nunique", "path": "count"})
    include_datasets = ds_counts.index[ds_counts["sub"] >= SUBS_PER_DATASET].values
    print(f"num include datasets: {len(include_datasets)}")

    shuffle_ds_ids = rng.permutation(len(include_datasets))
    split_idx = round(len(include_datasets) * TRAIN_FRACTION)
    dataset_splits = {
        "train": sorted(include_datasets[shuffle_ds_ids[:split_idx]]),
        "test": sorted(include_datasets[shuffle_ds_ids[split_idx:]]),
    }
    print("dataset splits:\n", json.dumps(dataset_splits, indent=4))

    file_splits = {}
    for split, split_datasets in dataset_splits.items():
        file_list = []
        for dataset in split_datasets:
            dsdf = df.loc[df["dataset"] == dataset]
            subs = dsdf["sub"].unique()
            subs = np.sort(rng.choice(subs, SUBS_PER_DATASET, replace=False))
            dsdf = dsdf.loc[dsdf["sub"].isin(subs)]
            for _, subdf in dsdf.groupby("sub"):
                path = rng.choice(subdf["path"])
                path = str(Path(path).relative_to(root))
                file_list.append(path)
        file_splits[split] = file_list

    out_path = ROOT / "config/openneuro_file_splits.json"
    with out_path.open("w") as f:
        json.dump(file_splits, f, indent=4)


def parse_metadata(path: Path):
    dataset = next(part for part in path.parts if part.startswith("ds"))
    dataset, _ = dataset.split("-")
    meta = dict(item.split("-") for item in path.stem.split("_") if "-" in item)
    meta = {k: meta.get(k) for k in ["sub", "ses", "task", "run", "acq"]}
    if meta["run"] is not None:
        meta["run"] = int(meta["run"])
    meta = {"dataset": dataset, **meta}
    return meta


if __name__ == "__main__":
    main()
