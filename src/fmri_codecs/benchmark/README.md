# fMRI compression benchmark

## Benchmark dataset construction

### 1. Download preprocessed OpenNeuro datasets

```bash
out_dir="datasets/openneuro"
mkdir -p "${out_dir}" 2>/dev/null

aws s3 sync --no-sign-request s3://openneuro-derivatives/fmriprep "${out_dir}" \
    --exclude "*" \
    --include "*_space-fsLR_den-91k_bold.dtseries.nii"
```

### 2. Generate dataset splits

As of 2025-11-11, there are 71 preprocessed datasets available at `s3://openneuro-derivatives` with a total of 9373 fMRI runs.
We include datasets with total number of runs between 20 and 200, and then split datasets into train/test so that each split contains roughly equal number of runs.

```bash
uv run python src/fmri_codecs/benchmark/make_openneuro_splits.py
```

### 3. Generate the huggingface dataset

To generate and upload the huggingface benchmark dataset, run

```bash
 uv run python src/fmri_codecs/benchmark/make_openneuro_dataset.py
```

This creates a dataset with the following [features](https://huggingface.co/docs/datasets/en/about_dataset_features).

```python
{
    "dataset": Value("string"),
    "sub": Value("string"),
    "ses": Value("string"),
    "task": Value("string"),
    "run": Value("int32"),
    "acq": Value("string"),
    "bold": Array2D(shape=(None, NUM_VERTICES), dtype="float16"),
}
```

To update the dataset splits, copy the output from [Step 2](#2-generate-dataset-splits) into the dataset generation script.
