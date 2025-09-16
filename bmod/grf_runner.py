from __future__ import annotations
# from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import logging
import pandas as pd
import os
# import numpy as np


logger = logging.getLogger(__name__)


def load_giraffe_csv(input_file: Path) -> pd.DataFrame:
    """
    Load Giraffe CSV file and return a DataFrame with depth and gain data.
    Note, the CSV file has a non-standard format, so we need to parse it manually.
    """
    depths, gains = [], []
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("Curve depth"):
                depths = [float(x) for x in next(f).strip().split(";") if x]
            if line.startswith("Curve gains"):
                gains = [float(x) for x in next(f).strip().split(";") if x]
    if not depths or not gains:
        raise ValueError("Could not find curve data in file")
    return pd.DataFrame({"depth_mm": depths, "gain_counts": gains})


def load_giraffe_dir(input_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Loop through all .csv files in a directory and parse them.
    Returns a dictionary: {filename: dataframe}.
    """
    results = {}
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(".csv"):
            fpath = os.path.join(input_dir, fname)
            try:
                results[fname] = load_giraffe_csv(fpath)
            except Exception as e:
                print(f"⚠️ Skipping {fname}: {e}")
    return results


def run(input_dir: Path,
        cfg: Dict[str, Any] | Any,
        write: Optional[Path] = None) -> pd.DataFrame:
    """
    Top-level pipeline for Giraffe processing.
    """
    logger.info("Giraffe runner started: %s", input_dir)

    # --- config bits
    gcfg = cfg.get("giraffe", {})

    # --- load data
    if not input_dir.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_dir}")
    if input_dir.is_file():
        data = {input_dir.name: load_giraffe_csv(input_dir)}
    elif input_dir.is_dir():
        data = load_giraffe_dir(input_dir)
    else:
        raise ValueError(f"Input path is neither a file nor a directory: {input_dir}")

    # log debug list of energies and wet value
    energies = gcfg.get("energies", [])
    logger.debug(f"Configured energies: {energies}")
    wet = gcfg.get("wet", 1.125)
    logger.debug(f"Configured water equivalent thickness (wet): {wet} cm")

    # list all data dict:
    for fname, df in data.items():
        logger.info(f"Processing file: {fname} with {len(df)} data points")
        # placeholder for actual processing logic
        # e.g., calibrate gains, fit curves, etc.
        # For now, just log the first few rows
        logger.debug(f"Data preview:\n{df.head()}")
