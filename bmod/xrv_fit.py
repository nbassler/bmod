from __future__ import annotations
import numpy as np
from scipy.optimize import curve_fit


def _gauss2d(coords, amp, x0, y0, sx, sy, theta, offset):
    x, y = coords
    ct, st = np.cos(theta), np.sin(theta)
    xr = ct * (x - x0) + st * (y - y0)
    yr = -st * (x - x0) + ct * (y - y0)
    return amp * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2)) + offset


def fit_gaussian2d(img: np.ndarray, p0: dict | None = None) -> dict:
    h, w = img.shape
    y, x = np.mgrid[0:h, 0:w]

    # Use initial guess or fallback
    if p0 is None:
        amp = float(img.max() - img.min())
        offset = float(np.median(img))
        x0 = float(np.argmax(img) % w)
        y0 = float(np.argmax(img) // w)
        sx = w / 6.0
        sy = h / 6.0
        theta = 0.0
    else:
        amp = p0["amp"]
        x0 = p0["x0"]
        y0 = p0["y0"]
        sx = p0["sigma_x"]
        sy = p0["sigma_y"]
        theta = p0.get("theta", 0.0)
        offset = p0["offset"]

    # Normalize image to [0, 1] for numerical stability
    img_normalized = (img - offset) / amp
    p0_vec_normalized = [1.0, x0, y0, sx, sy, theta, 0.0]

    # Define bounds (normalized)
    bounds = (
        [0.5, x0-10, y0-10, 1.0, 1.0, theta-np.pi/4, -0.1],
        [1.5, x0+10, y0+10, 2*sx, 2*sy, theta+np.pi/4, 0.1],
    )

    try:
        popt, _ = curve_fit(
            _gauss2d,  # Takes (x_flat, y_flat), returns flattened array
            (x.ravel(), y.ravel()),  # Pass flattened coordinates
            img_normalized.ravel(),  # Pass flattened image data
            p0=p0_vec_normalized,
            bounds=bounds,
            maxfev=10000,
        )
        # Rescale results
        amp_fit, x0_fit, y0_fit, sx_fit, sy_fit, theta_fit, offset_fit = popt
        amp_fit *= amp
        offset_fit = offset_fit * amp + offset

        # Reshape the fitted model for residuals/R²
        fit = _gauss2d((x, y), amp_fit, x0_fit, y0_fit, sx_fit, sy_fit, theta_fit, offset_fit).reshape(h, w)
        resid = img - fit  # Both are (1200, 1600)
        rss = float((resid**2).sum())
        tss = float(((img - img.mean())**2).sum())
        r2 = 1.0 - rss / tss if tss > 0 else np.nan

        return dict(
            amp=amp_fit, x0=x0_fit, y0=y0_fit, sigma_x=sx_fit, sigma_y=sy_fit,
            theta=theta_fit, offset=offset_fit, r2=r2, rss=rss, success=True
        )
    except Exception as e:
        print(f"Fit failed: {e}")
        return dict(
            amp=np.nan, x0=np.nan, y0=np.nan, sigma_x=np.nan, sigma_y=np.nan,
            theta=np.nan, offset=np.nan, r2=np.nan, rss=np.nan, success=False
        )
