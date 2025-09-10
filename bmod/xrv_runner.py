from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import logging
import pandas as pd
# import numpy as np

from bmod import xrv_io
from bmod import xrv_guess
from bmod import xrv_fit


logger = logging.getLogger(__name__)


@dataclass
class XRVResult:
    file: Path
    z: Optional[float]  # fill from folder name or config logic (optional)
    amp: float
    x0: float
    y0: float
    sigma_x: float
    sigma_y: float
    theta: float
    offset: float
    # Quality metrics
    r2: float
    rss: float
    success: bool


def run(input_dir: Path,
        cfg: Dict[str, Any] | Any,
        write: Optional[Path] = None) -> pd.DataFrame:
    """
    Top-level pipeline for XRV processing.
    """
    logger.info("XRV runner started: %s", input_dir)

    # --- config bits
    xcfg = cfg.get("xrv", {})
    zpos = xcfg.get("zpos")
    if not isinstance(zpos, (list, tuple)):
        raise ValueError("xrv.zpos must be a list of floats")
    logger.info("Using z positions (n=%d): %s", len(zpos), zpos)

    origin = tuple(xcfg.get("origin", (0.0, 0.0)))
    scaling = tuple(xcfg.get("scaling", (1.0, 1.0)))
    sx_mm_per_px, sy_mm_per_px = scaling

    # --- discover files
    tiffs = sorted(xrv_io.find_tiffs(input_dir))
    if not tiffs:
        logger.warning("No TIFFs found under %s", input_dir)
        return pd.DataFrame()  # empty

    rows: List[Dict[str, Any]] = []

    # --- process
    for i, tpath in enumerate(tiffs):
        img = xrv_io.load_tiff(tpath)

        guess = xrv_guess.initial_guess_single_spot(
            img,
            median_size=xcfg.get("median_size", 3),
            snr=xcfg.get("min_snr", 5.0),
            window_radius=xcfg.get("window_radius", 10),
        )
        fit = xrv_fit.fit_gaussian2d(img, p0=guess)

        # pixel → mm
        x_mm = (fit["x0"] - origin[0]) * sx_mm_per_px
        y_mm = (fit["y0"] - origin[1]) * sy_mm_per_px
        sx_mm = (fit["sigma_x"]) * sx_mm_per_px
        sy_mm = (fit["sigma_y"]) * sy_mm_per_px

        # choose z (simple 1:1 mapping; refine later if needed)
        z = zpos[i] if i < len(zpos) else None

        # build one row (pixels + mm)
        row = {
            "file": str(tpath),
            "z": z,
            # pixels
            "amp": fit["amp"],
            "x0_px": fit["x0"],
            "y0_px": fit["y0"],
            "sigma_x_px": fit["sigma_x"],
            "sigma_y_px": fit["sigma_y"],
            "theta_rad": fit["theta"],
            "offset": fit["offset"],
            "r2": fit["r2"],
            "rss": fit["rss"],
            "success": fit["success"],
            # mm (calibrated)
            "x0_mm": x_mm,
            "y0_mm": y_mm,
            "sigma_x_mm": sx_mm,
            "sigma_y_mm": sy_mm,
        }
        rows.append(row)

    # --- to DataFrame
    df = pd.DataFrame(rows)

    # --- optional write
    if write:
        out = Path(write)
        if out.suffix.lower() == ".parquet":
            df.to_parquet(out, index=False)
        else:
            df.to_csv(out, index=False)
        logger.info("Wrote results -> %s", out)

    return df
