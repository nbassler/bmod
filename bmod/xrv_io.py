"""
xrv_io.py

I/O utilities for XRV analysis.
Responsible for:
- scanning directories for TIFF files
- loading TIFF images into numpy arrays
- optional grouping of files by subdirectory/session
"""

from pathlib import Path
from typing import Iterator, List

import numpy as np
from tifffile import imread   # or from PIL import Image


logger = __import__('logging').getLogger(__name__)


def find_tiffs(root: Path, extensions: List[str] = [".tif", ".tiff"]) -> Iterator[Path]:
    """
    Recursively find all TIFF files under the given root directory.

    Parameters
    ----------
    root : Path
        The input directory to scan.
    extensions : list of str
        File extensions to match (default: .tif, .tiff).

    Yields
    ------
    Path
        Path to each TIFF file discovered.
    """
    for path in root.rglob("*"):
        if path.suffix.lower() in extensions:
            yield path


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
