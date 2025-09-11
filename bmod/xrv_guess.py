from __future__ import annotations

import logging
import numpy as np

from scipy.ndimage import median_filter, label, center_of_mass


logger = logging.getLogger(__name__)


def initial_guess_single_spot(
    img: np.ndarray,
    *,
    median_size: int = 3,
    window_radius: int = 20,
):
    """ Initial guess for a single bright spot in the image.
    Parameters
    ----------
    img : np.ndarray
        2D image array.
    median_size : int
        Size of the median filter window.
    window_radius : int
        Radius of the local window for size estimation. Should be larger than expected spot size.
    Returns
    -------
    dict
        Dictionary with initial guess parameters.
    """
    # 0. Preprocess: median filter to reduce noise
    work = img.astype(np.float32, copy=True)
    if median_size and median_size > 1:
        work = median_filter(work, size=median_size)

    # 1. Use half of the max value as the threshold
    thr = 0.5 * work.max()

    # 2. Create mask and select largest blob
    mask = work >= thr
    labels, n = label(mask)

    if n == 0:
        y0, x0 = np.unravel_index(np.argmax(work), work.shape)
        print("Warning: No spot found above threshold; using global max.", y0, x0)
    else:
        areas = [(labels == k).sum() for k in range(1, n + 1)]
        kmax = 1 + int(np.argmax(areas))
        cy, cx = center_of_mass(mask, labels=labels, index=kmax)
        y0, x0 = int(round(cy)), int(round(cx))
        print(f"Found {n} spots; using largest at ({y0}, {x0}) with area {areas[kmax-1]}.")

    # 3. Background: median of non-spot pixels
    bg_mask = work < thr
    med = np.median(work[bg_mask])
    offset = float(med)

    # 4. Amplitude and size
    amp = float(work[y0, x0] - med)

    # 5. Local window for size estimation
    y0a = max(0, y0 - window_radius)
    y1a = min(work.shape[0], y0 + window_radius + 1)
    x0a = max(0, x0 - window_radius)
    x1a = min(work.shape[1], x0 + window_radius + 1)

    # Extract the patch and its local coordinates
    patch = work[y0a:y1a, x0a:x1a] - med
    patch[patch < 0] = 0

    if patch.sum() > 0:
        # Create local coordinates centered on (0, 0) for the patch
        yy_local, xx_local = np.mgrid[-(y0 - y0a):(y1a - y0), -(x0 - x0a):(x1a - x0)]
        wy = (patch * (yy_local) ** 2).sum() / patch.sum()
        wx = (patch * (xx_local) ** 2).sum() / patch.sum()
        sigma_y = float(np.sqrt(max(wy, 1e-12)))
        sigma_x = float(np.sqrt(max(wx, 1e-12)))
    else:
        sigma_x = sigma_y = 15.0  # Default for ~30-pixel FWHM

    # for debugging:
    # import matplotlib.pyplot as plt
    # logger.debug("Median:", med)
    # logger.debug("Threshold:", thr)
    # logger.debug("Max value:", work.max())
    # # After running initial_guess_single_spot:
    # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # axes[0].imshow(work, cmap="gray")
    # axes[0].axvline(x0, color="red", linestyle="--")
    # axes[0].axhline(y0, color="red", linestyle="--")
    # axes[0].set_title("Filtered image with centroid")
    # axes[1].imshow(mask, cmap="gray")
    # axes[1].set_title("Thresholded mask")
    # axes[2].imshow(patch, cmap="gray")
    # axes[2].set_title("Local patch for size estimation")
    # plt.show()

    return {
        "amp": max(amp, 1e-6),
        "x0": float(x0),
        "y0": float(y0),
        "sigma_x": float(sigma_x),
        "sigma_y": float(sigma_y),
        "theta": 0.0,        # rotation unnecessary as a start
        "offset": offset,
    }
