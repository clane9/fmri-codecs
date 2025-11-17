# fMRI codecs datasets

## OpenNeuro benchmark dataset construction

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
We include datasets with at least 20 subjects, and include runs with between 100 and 300 TRs. This leaves 30 datasets after filtering. We randomly split the datasets into a 50/50 train/test split. For each dataset, we include 20 random subjects and one run per subject.

```bash
uv run python src/fmri_codecs/benchmark/make_openneuro_splits.py
```

### 3. Generate the huggingface dataset

To generate the huggingface benchmark dataset, run

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

### 4. Upload the dataset to huggingface hub

For reproducibility, we upload the dataset to the huggingface hub at [`clane9/openneuro-fslr64k.arrow`](https://huggingface.co/datasets/clane9/openneuro-fslr64k.arrow).

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_large_folder(
    "clane9/openneuro-fslr64k.arrow",
    "datasets/openneuro-fslr64k.arrow",
    repo_type="dataset",
    num_workers=8,
)
```
