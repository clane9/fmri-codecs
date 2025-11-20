import argparse
import logging
import time
import yaml
from importlib import resources
from pathlib import Path

import datasets as hfds
import numpy as np
import pandas as pd
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

import fmri_codecs
import fmri_codecs.config
from fmri_codecs.utils import random_seed, setup_logger, encode_numpy


logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.WARNING,
    datefmt="%y-%m-%d %H:%M:%S",
)
_logger = logging.getLogger(__name__)

VMAX = 5.0

BENCH_DATASET = "clane9/openneuro-fslr64k.arrow"


def main(cfg: DictConfig):
    random_seed(cfg.seed)

    out_dir = Path(cfg.out_dir)
    num_runs = len(list(out_dir.glob("v*")))
    out_dir = out_dir / f"v{num_runs:03d}"
    out_dir.mkdir(parents=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")

    setup_logger(_logger, level=cfg.log_level, log_path=out_dir / "log.txt")
    _logger.info("Benchmarking fMRI codecs")
    _logger.info("Config:\n%s", yaml.safe_dump(OmegaConf.to_object(cfg), sort_keys=False))
    _logger.info(f"Saving to output dir: {out_dir}")

    modules = fmri_codecs.import_codecs()
    _logger.info(f"Found {len(modules)} codec modules:\n{modules}")

    _logger.info(f"Loading benchmark dataset: {BENCH_DATASET}")
    dataset_dict = hfds.load_dataset(BENCH_DATASET)
    train_dataset = dataset_dict["train"].with_format("numpy")
    test_dataset = dataset_dict["test"].with_format("numpy")

    names = list(cfg.codecs) if cfg.codecs else fmri_codecs.list_codecs()
    _logger.info(f"Loading codecs: {names}")
    codecs = {name: fmri_codecs.create_codec(name) for name in names}

    results = []
    for name, codec in codecs.items():
        if hasattr(codec, "fit"):
            _logger.info(f"Fitting {name}")
            fit_single(codec, train_dataset, vmax=VMAX)

        _logger.info(f"Evaluating {name}")
        result = evaluate_single(codec, test_dataset, vmax=VMAX)
        result.insert(0, "codec", name)

        _logger.info(f"Result ({name}):\n{result.to_markdown(index=False)}")
        result.to_csv(out_dir / f"result__{name}.csv", index=False)
        results.append(result)

    results = pd.concat(results, axis=0, ignore_index=True)
    summary = (
        results.groupby("codec")
        .agg(
            {
                "decode_fps": ["mean", "std"],
                "log_ratio": ["mean", "std"],
                "psnr": ["mean", "std"],
            }
        )
        .reset_index()
    )
    summary.columns = ["codec"] + [
        f"{metric}__{agg}"
        for metric in ["decode_fps", "log_ratio", "psnr"]
        for agg in ["mean", "std"]
    ]
    _logger.info(f"Summary:\n{summary.to_markdown(index=False)}")
    summary.to_csv(out_dir / "summary.csv", index=False)


def fit_single(
    codec: fmri_codecs.Codec,
    train_dataset: hfds.Dataset,
    vmax: float = 5.0,
):
    X_train = [clip_values(sample["bold"], vmax) for sample in tqdm(train_dataset)]
    codec.fit(X_train)
    return codec


def evaluate_single(
    codec: fmri_codecs.Codec,
    test_dataset: hfds.Dataset,
    vmax: float = 5.0,
):
    records = []
    for sample in tqdm(test_dataset):
        x = clip_values(sample.pop("bold"), vmax)
        buf = codec.encode(x)
        buf_raw = encode_numpy(x.astype(np.float16))

        tic = time.perf_counter()
        x_ = codec.decode(buf)
        dec_t = time.perf_counter() - tic

        err = np.sum((x - x_) ** 2)

        record = {
            **sample,
            "decode_time": dec_t,
            "length": len(buf),
            "length_raw": len(buf_raw),
            "err": err,
        }
        records.append(record)

    table = pd.DataFrame.from_records(records)
    result = (
        table.groupby("dataset")
        .agg(
            {
                "n_frames": "sum",
                "decode_time": "sum",
                "length": "sum",
                "length_raw": "sum",
                "err": "sum",
            }
        )
        .reset_index()
    )
    result["decode_fps"] = result["n_frames"] / result["decode_time"]
    result["log_ratio"] = np.log10(result["length"] / result["length_raw"])
    mse = result["err"] / result["n_frames"]
    result["psnr"] = 10 * np.log10(((2 * vmax) ** 2) / mse)

    result = result.loc[:, ["dataset", "decode_fps", "log_ratio", "psnr"]]
    return result


def clip_values(x: np.ndarray, vmax: float) -> np.ndarray:
    return np.clip(x, -vmax, vmax)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default=None)
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    args = parser.parse_args(
        [
            "--overrides",
            "codecs=[bpeg_n400_d32_nb1024_none]",
        ]
    )

    with resources.path(fmri_codecs.config, "default_benchmark.yaml") as default_cfg_path:
        cfg = OmegaConf.load(default_cfg_path)
    if args.cfg_path:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(args.cfg_path))
    if args.overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(args.overrides))
    main(cfg)
