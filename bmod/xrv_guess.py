from __future__ import annotations
import numpy as np
from scipy.ndimage import median_filter, label, center_of_mass


def initial_guess_single_spot(
    img: np.ndarray,
    *,
    median_size: int = 3,
    snr: float = 5.0,
    window_radius: int = 10,
):
    """
    Fast, robust initial guess for one main spot:
    1) median-filter to kill hot pixels
    2) robust background/noise (median + MAD)
    3) threshold -> largest blob -> centroid
    4) rough size from windowed second moments
    """
    work = img.astype(np.float32, copy=False)
    if median_size and median_size > 1:
        work = median_filter(work, size=median_size)

    # robust background + noise
    med = np.median(work)
    mad = np.median(np.abs(work - med)) + 1e-12
    sigma = 1.4826 * mad

    thr = med + snr * max(sigma, 1e-6)
    mask = work >= thr

    # pick largest blob (or fallback to global max)
    labels, n = label(mask)
    if n == 0:
        y0, x0 = np.unravel_index(np.argmax(work), work.shape)
    else:
        areas = [(labels == k).sum() for k in range(1, n + 1)]
        kmax = 1 + int(np.argmax(areas))
        cy, cx = center_of_mass(mask, labels=labels, index=kmax)
        y0, x0 = int(round(cy)), int(round(cx))

    # amplitude / offset
    amp = float(work[y0, x0] - med)
    offset = float(med)

    # crude size from local window moments
    y, x = np.mgrid[0:work.shape[0], 0:work.shape[1]]
    y0a = max(0, y0 - window_radius)
    y1a = min(work.shape[0], y0 + window_radius + 1)
    x0a = max(0, x0 - window_radius)
    x1a = min(work.shape[1], x0 + window_radius + 1)
    patch = work[y0a:y1a, x0a:x1a] - med
    patch[patch < 0] = 0
    if patch.sum() > 0:
        yy, xx = np.mgrid[y0a:y1a, x0a:x1a]
        wy = (patch * (yy - y0) ** 2).sum() / patch.sum()
        wx = (patch * (xx - x0) ** 2).sum() / patch.sum()
        sigma_y = float(np.sqrt(max(wy, 1e-12)))
        sigma_x = float(np.sqrt(max(wx, 1e-12)))
    else:
        sigma_x = sigma_y = 3.0  # conservative default

    return {
        "amp": max(amp, 1e-6),
        "x0": float(x0),
        "y0": float(y0),
        "sigma_x": float(sigma_x),
        "sigma_y": float(sigma_y),
        "theta": 0.0,        # rotation unnecessary as a start
        "offset": offset,
    }
