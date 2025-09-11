"""
xrv_io.py

I/O utilities for XRV analysis.
Responsible for:
- scanning directories for TIFF files
- loading TIFF images into numpy arrays
- optional grouping of files by subdirectory/session
"""

from pathlib import Path
from typing import List
import re
from typing import Dict

import numpy as np
from tifffile import imread   # or from PIL import Image


logger = __import__('logging').getLogger(__name__)


def natural_key(path: Path) -> List:
    """
    Sort key that treats digits as numbers (for '1', '2', ..., '10').
    """
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.stem)]


def find_tiffs_by_z(root: Path,
                    extensions: List[str] = [".tif", ".tiff"]) -> Dict[Path, List[Path]]:
    """
    Discover TIFF files grouped by z-directory.

    Parameters
    ----------
    root : Path
        Root input directory containing subdirectories for each z-position.
    extensions : list of str
        File extensions to match.

    Returns
    -------
    dict
        Mapping: z-directory -> list of sorted TIFF paths inside.
    """
    groups: Dict[Path, List[Path]] = {}

    for zdir in sorted([p for p in root.iterdir() if p.is_dir()], key=natural_key):
        tiffs = [p for p in zdir.iterdir() if p.suffix.lower() in extensions]
        tiffs.sort(key=natural_key)
        if tiffs:
            groups[zdir] = tiffs

    return groups


def load_tiff(path: Path) -> np.ndarray:
    """
    Load a TIFF file into a NumPy array.

    Parameters
    ----------
    path : Path
        Path to the TIFF file.

    Returns
    -------
    np.ndarray
        Image data as a NumPy array.
    """
    logger.info(f"Loading TIFF file: {path}")

    img = imread(path)
    return img.astype(np.float32)   # float32 to make downstream processing consistent


def group_by_parent(paths: List[Path]) -> dict[str, List[Path]]:
    """
    Group TIFF files by their immediate parent directory.

    Parameters
    ----------
    paths : list of Path
        TIFF file paths.

    Returns
    -------
    dict
        Mapping: parent directory name -> list of TIFF paths.
    """
    groups = {}
    for p in paths:
        groups.setdefault(p.parent.name, []).append(p)
    return groups
