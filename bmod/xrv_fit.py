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

    if p0 is None:
        # worst-case fallback
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

    p0_vec = [amp, x0, y0, sx, sy, theta, offset]

    # Define bounds
    bounds = (
        # Lower bounds: amp (free), x0-10, y0-10, sx-10, sy-10, theta-π/4, offset-0.1*offset
        [0, x0-10, y0-10, sx-10, sy-10, theta-np.pi/4, offset-0.1*abs(offset)],
        # Upper bounds: amp (free), x0+10, y0+10, sx+10, sy+10, theta+π/4, offset+0.1*abs(offset)
        [np.inf, x0+10, y0+10, sx+10, sy+10, theta+np.pi/4, offset+0.1*abs(offset)]
    )

    try:
        popt, _ = curve_fit(_gauss2d, (x, y), img.ravel(), p0=p0_vec, maxfev=10000, bounds=bounds)
        fit = _gauss2d((x, y), *popt).reshape(h, w)
        resid = img - fit
        rss = float((resid**2).sum())
        tss = float(((img - img.mean())**2).sum())
        r2 = 1.0 - rss / tss if tss > 0 else np.nan
        amp, x0, y0, sx, sy, theta, offset = map(float, popt)
        return dict(amp=amp, x0=x0, y0=y0, sigma_x=sx, sigma_y=sy, theta=theta,
                    offset=offset, r2=r2, rss=rss, success=True)
    except Exception:
        return dict(amp=np.nan, x0=np.nan, y0=np.nan, sigma_x=np.nan, sigma_y=np.nan,
                    theta=np.nan, offset=np.nan, r2=np.nan, rss=np.nan, success=False)
