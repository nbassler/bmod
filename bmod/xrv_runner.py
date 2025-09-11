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
    z: float
    energy: float
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

    energies = xcfg.get("energies")
    if not isinstance(energies, (list, tuple)):
        raise ValueError("xrv.energies must be a list of floats")

    window_radius = xcfg.get("window_radius", 50)
    if not isinstance(window_radius, int) or window_radius <= 0:
        raise ValueError("xrv.window_radius must be a positive integer")

    origin = tuple(xcfg.get("origin", (0.0, 0.0)))
    scaling = tuple(xcfg.get("scaling", (1.0, 1.0)))
    sx_px_per_mm, sy_px_per_mm = scaling

    groups = xrv_io.find_tiffs_by_z(input_dir)

    i = 0
    # just a check for what files are opened:
    # for zdir, tiffs in groups.items():
    #     z = zpos[i] if i < len(zpos) else None
    #     i += 1
    #     logger.info("Z dir: %s with %d files", zdir.name, len(tiffs))
    #     for t in tiffs:
    #         logger.info("   %s", t.name)

    rows: List[Dict[str, Any]] = []

    # --- process
    i = 0
    for zdir, tiffs in groups.items():
        # choose z (deterministic as listed in input config file)
        if i < len(zpos):
            z = zpos[i]
        else:
            raise ValueError("More z-directories than z-positions")

        for j, tpath in enumerate(tiffs):
            logger.info("Processing z=%.3f file %d/%d: %s", z, j+1, len(tiffs), tpath.name)
            if j < len(energies):
                energy = energies[j]
            else:
                raise ValueError("More files per z than energies")

            img = xrv_io.load_tiff(tpath)

            guess = xrv_guess.initial_guess_single_spot(
                img,
                median_size=xcfg.get("median_size", 3),
                window_radius=xcfg.get("window_radius", 30),
            )
            logger.debug("Initial guess: %s", guess)

            fit = xrv_fit.fit_gaussian2d(img, p0=guess, window_radius=window_radius)

            logger.info("Fit result: %s", fit)
            if fit.get("success") is not True:
                logger.warning("Fit failed for %s", tpath)
                return None

            # pixel → mm
            x_mm = (fit["x0"] - origin[0]) / sx_px_per_mm
            y_mm = (fit["y0"] - origin[1]) / sy_px_per_mm
            sx_mm = (fit["sigma_x"]) / sx_px_per_mm
            sy_mm = (fit["sigma_y"]) / sy_px_per_mm

            # build one row (pixels + mm)
            row = {
                "file": str(tpath),
                "z": z,
                "energy": energy,
                # pixels
                "amp": fit["amp"],
                "x0_px": fit["x0"],
                "y0_px": fit["y0"],
                "sigma_x_px": fit["sigma_x"],
                "sigma_y_px": fit["sigma_y"],
                "theta_deg": fit["theta"] * 180.0 / 3.141592653589793,
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
        i += 1

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
