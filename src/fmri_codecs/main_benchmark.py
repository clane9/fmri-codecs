import argparse
import json
import logging
import random
import sys
import time
import yaml
from importlib import resources
from itertools import product
from pathlib import Path

import datasets as hfds
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

import fmri_codecs.config
from fmri_codecs.codecs.base import Codec
from fmri_codecs.codecs.common import encode_numpy
from fmri_codecs.codecs.registry import create_codec, import_codecs


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

    modules = import_codecs()
    _logger.info(f"Found {len(modules)} codec modules:\n{modules}")

    _logger.info(f"Loading benchmark dataset: {BENCH_DATASET}")
    dataset_dict = hfds.load_dataset(BENCH_DATASET)
    train_dataset = dataset_dict["train"].with_format("numpy")
    test_dataset = dataset_dict["test"].with_format("numpy")

    names = list(cfg.include_codecs) if cfg.include_codecs else list(cfg.codecs)
    _logger.info(f"Loading codecs: {names}")

    codecs: list[tuple[str, Codec]] = []
    for name in names:
        hparam_grid = cfg.codecs[name]
        for hparams in generate_hparams(hparam_grid):
            codec = create_codec(name, **hparams)
            codecs.append((name, codec))

    codecs_fmt = "\n".join(f"- {codec}" for _, codec in codecs)
    _logger.info(f"\n{codecs_fmt}")

    results = []
    for name, codec in codecs:
        if hasattr(codec, "fit"):
            _logger.info(f"Fitting {name}")
            fit_single(codec, train_dataset, vmax=VMAX)

        _logger.info(f"Evaluating {name}")
        result = evaluate_single(codec, test_dataset, vmax=VMAX)
        result.insert(0, "hparams", json.dumps(codec.hparams()))
        result.insert(0, "name", name)
        result.insert(0, "codec", str(codec))

        _logger.info(f"Result ({codec}):\n{result.to_markdown(index=False)}")
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


def generate_hparams(hparam_grid: dict[str, list[int | str | float]]):
    param_values = product(*hparam_grid.values())
    for values in param_values:
        hparams = {k: v for k, v in zip(hparam_grid, values)}
        yield hparams


def fit_single(
    codec: Codec,
    train_dataset: hfds.Dataset,
    vmax: float = 5.0,
):
    X_train = [clip_values(sample["bold"], vmax) for sample in tqdm(train_dataset)]
    codec.fit(X_train)
    return codec


def evaluate_single(
    codec: Codec,
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

        err = ((x - x_) ** 2).mean(axis=-1).sum()

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
    result["psnr"] = 10 * np.log10(1 / mse)  # inputs are std scaled so baseline mse is 1

    result = result.loc[:, ["dataset", "decode_fps", "log_ratio", "psnr"]]
    return result


def random_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_logger(
    logger: logging.Logger,
    level: str = "INFO",
    log_path: Path | None = None,
):
    logger.setLevel(level)

    # clean up any existing handlers
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.root.handlers = []

    fmt = "[%(levelname)s %(asctime)s]: %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%y-%m-%d %H:%M:%S")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(level)
    logger.addHandler(handler)

    if log_path:
        handler = logging.FileHandler(log_path)
        handler.setFormatter(formatter)
        handler.setLevel(level)
        logger.addHandler(handler)


def clip_values(x: np.ndarray, vmax: float) -> np.ndarray:
    return np.clip(x, -vmax, vmax)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default=None)
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    args = parser.parse_args()

    with resources.path(fmri_codecs.config, "default_benchmark.yaml") as default_cfg_path:
        cfg = OmegaConf.load(default_cfg_path)
    if args.cfg_path:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(args.cfg_path))
    if args.overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(args.overrides))
    main(cfg)
