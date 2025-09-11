from __future__ import annotations
import numpy as np
from scipy.optimize import curve_fit


def _gauss2d(coords, amp, x0, y0, sx, sy, ct, st, offset):
    x, y = coords
    xr = ct * (x - x0) + st * (y - y0)
    yr = -st * (x - x0) + ct * (y - y0)
    return amp * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2)) + offset


def fit_gaussian2d(img_full: np.ndarray, p0: dict | None = None, window_radius=50) -> dict:
    # Crop to local window around initial guess
    x0_full, y0_full = p0["x0"], p0["y0"]
    y_slice = slice(max(0, int(y0_full) - window_radius), min(img_full.shape[0], int(y0_full) + window_radius + 1))
    x_slice = slice(max(0, int(x0_full) - window_radius), min(img_full.shape[1], int(x0_full) + window_radius + 1))
    img = img_full[y_slice, x_slice]

    # Create local coordinates for the cropped window
    h, w = img.shape
    y_local, x_local = np.mgrid[0:h, 0:w]

    # Adjust initial guess to local coordinates
    x0_local = x0_full - x_slice.start
    y0_local = y0_full - y_slice.start

    # Use initial guess (local coordinates)
    amp = p0["amp"]
    sx = p0["sigma_x"]
    sy = p0["sigma_y"]
    theta = p0.get("theta", 0.0)
    ct, st = np.cos(theta), np.sin(theta)
    offset = p0["offset"]

    # Normalize image to [0, 1] for numerical stability
    img_normalized = (img - offset) / amp
    p0_vec_normalized = [1.0, x0_local, y0_local, sx, sy, ct, st, 0.0]

    # Define bounds (normalized, local coordinates)
    bounds = (
        [0.5, x0_local-10, y0_local-10, 1.0, 1.0, -1, -1, -0.1],  # Lower bounds for ct, st
        [1.5, x0_local+10, y0_local+10, 2*sx, 2*sy, 1, 1, 0.1],   # Upper bounds for ct, st
    )

    try:
        popt, _ = curve_fit(
            lambda coords, amp, x0, y0, sx, sy, ct, st, offset:
                _gauss2d(coords, amp, x0, y0, sx, sy, ct, st, offset),
            (x_local.ravel(), y_local.ravel()),  # Local coordinates
            img_normalized.ravel(),
            p0=p0_vec_normalized,
            bounds=bounds,
            maxfev=10000,
        )

        # Rescale results
        amp_fit, x0_fit_local, y0_fit_local, sx_fit, sy_fit, ct_fit, st_fit, offset_fit = popt
        amp_fit *= amp
        offset_fit = offset_fit * amp + offset

        # Reshape the fitted model for residuals/R² (local coordinates)
        fit = _gauss2d((x_local, y_local), amp_fit, x0_fit_local, y0_fit_local,
                       sx_fit, sy_fit, ct_fit, st_fit, offset_fit).reshape(h, w)
        resid = img - fit
        rss = float((resid**2).sum())
        tss = float(((img - img.mean())**2).sum())
        r2 = 1.0 - rss / tss if tss > 0 else np.nan

        # Convert fitted coordinates back to full image coordinates
        x0_fit = x0_fit_local + x_slice.start
        y0_fit = y0_fit_local + y_slice.start

        return dict(
            amp=amp_fit, x0=x0_fit, y0=y0_fit, sigma_x=sx_fit, sigma_y=sy_fit,
            theta=np.arctan2(st_fit, ct_fit), offset=offset_fit, r2=r2, rss=rss, success=True
        )
    except Exception as e:
        print(f"Fit failed: {e}")
        return dict(
            amp=np.nan, x0=np.nan, y0=np.nan, sigma_x=np.nan, sigma_y=np.nan,
            theta=np.nan, offset=np.nan, r2=np.nan, rss=np.nan, success=False
        )
