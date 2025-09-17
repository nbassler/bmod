from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)

# ----------------------------
# Spline basis infrastructure
# ----------------------------


def _normalize(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    scale = xmax - xmin if xmax > xmin else 1.0
    return (x - xmin) / scale, xmin, scale


def _make_bspline_basis(x: np.ndarray, degree: int = 3, n_bases: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    if n_bases < degree + 1:
        n_bases = degree + 1
    m = n_bases - degree - 1
    if m > 0:
        internal = np.linspace(0, 1, m + 2)[1:-1]
    else:
        internal = np.array([])
    t = np.concatenate([
        np.zeros(degree + 1),
        internal,
        np.ones(degree + 1)
    ])
    B = np.empty((x.size, n_bases), dtype=float)
    for j in range(n_bases):
        c = np.zeros(n_bases)
        c[j] = 1.0
        B[:, j] = BSpline(t, c, degree)(x)
    return B, t


def _second_diff_matrix(n: int) -> np.ndarray:
    if n < 3:
        return np.zeros((0, n))
    D = np.zeros((n - 2, n), dtype=float)
    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D

# ----------------------------
# Global model fit per plane
# ----------------------------


@dataclass(frozen=True)
class _PlaneFit:
    coef_A: np.ndarray
    coef_B: np.ndarray
    coef_C: np.ndarray
    coef_D: np.ndarray
    knots:   np.ndarray
    degree:  int
    E_min: float
    E_scale: float
    n_bases: int


def _fit_plane_global(
    df: pd.DataFrame,
    z0: float,
    plane: str,
    *,
    n_bases: int = 8,
    degree: int = 3,
    lambda_reg: Tuple[float, float, float, float] = (1e-2, 1e-2, 1e-2, 1e-2),
    weight_col: Optional[str] = None
) -> _PlaneFit:
    z = df["z"].to_numpy(dtype=float)
    E = df["energy"].to_numpy(dtype=float)
    sigma = df[f"sigma_{plane}_mm"].to_numpy(dtype=float)
    E_n, E_min, E_scale = _normalize(E)
    B, knots = _make_bspline_basis(E_n, degree=degree, n_bases=n_bases)
    K = B.shape[1]
    L = z - float(z0)
    Z2 = L**2
    Z1 = L
    Z0 = np.ones_like(L)
    Z3 = L**3
    X_blocks = [Z2[:, None] * B, Z1[:, None] * B, Z0[:, None] * B, Z3[:, None] * B]
    X = np.concatenate(X_blocks, axis=1)
    y = np.square(sigma)
    if weight_col is not None and weight_col in df.columns:
        w_raw = np.asarray(df[weight_col].to_numpy(dtype=float))
        w_raw = np.clip(w_raw, 1e-6, 1.0)
    else:
        w_raw = np.ones_like(y)
    w = np.sqrt(w_raw)
    D2 = _second_diff_matrix(K)
    blocks = []
    for lam in lambda_reg:
        if D2.size == 0 or lam <= 0:
            continue
        blocks.append(np.sqrt(lam) * np.pad(D2, ((0, 0), (0, 0))))
    if D2.size == 0:
        R = np.zeros((0, 4 * K))
    else:
        rows = []
        for i, lam in enumerate(lambda_reg):
            if lam <= 0:
                continue
            row = np.zeros((D2.shape[0], 4 * K))
            col0 = i * K
            row[:, col0:col0 + K] = np.sqrt(lam) * D2
            rows.append(row)
        R = np.vstack(rows) if rows else np.zeros((0, 4 * K))
    WX = (w[:, None] * X)
    Wy = (w * y)
    A = np.vstack([WX, R])
    b = np.concatenate([Wy, np.zeros(R.shape[0])])
    theta, *_ = np.linalg.lstsq(A, b, rcond=None)
    coef_A = theta[0*K:1*K]
    coef_B = theta[1*K:2*K]
    coef_C = theta[2*K:3*K]
    coef_D = theta[3*K:4*K]
    return _PlaneFit(
        coef_A=coef_A, coef_B=coef_B, coef_C=coef_C, coef_D=coef_D,
        knots=knots, degree=degree, E_min=E_min, E_scale=E_scale, n_bases=K
    )


def _eval_coeffs_at_energies(fit: _PlaneFit, energies: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    E_n = (energies - fit.E_min) / (fit.E_scale if fit.E_scale != 0 else 1.0)
    B = np.empty((E_n.size, fit.n_bases), dtype=float)
    for j in range(fit.n_bases):
        c = np.zeros(fit.n_bases)
        c[j] = 1.0
        B[:, j] = BSpline(fit.knots, c, fit.degree)(E_n)
    A = B @ fit.coef_A
    Bc = B @ fit.coef_B
    C = B @ fit.coef_C
    D = B @ fit.coef_D
    return A, Bc, C, D

# ----------------------------
# Public API (synced with quadratic)
# ----------------------------


def fit_all_energies(
    df: pd.DataFrame,
    z0: float = 0.0,
    z_prime: float = 0.0,
    zdir_negative: bool = True,
    *,
    n_bases: int = 8,
    degree: int = 3,
    lambda_reg_xy: Tuple[float, float, float, float] = (1e-2, 1e-2, 1e-2, 1e-2),
    weight_col: Optional[str] = None
) -> pd.DataFrame:
    required = ['z', 'sigma_x_mm', 'sigma_y_mm', 'energy']
    if not all(c in df.columns for c in required):
        raise ValueError(f"DataFrame must contain columns: {required}")
    fit_x = _fit_plane_global(df, z0, plane="x",
                              n_bases=n_bases, degree=degree,
                              lambda_reg=lambda_reg_xy, weight_col=weight_col)
    fit_y = _fit_plane_global(df, z0, plane="y",
                              n_bases=n_bases, degree=degree,
                              lambda_reg=lambda_reg_xy, weight_col=weight_col)
    energies = np.sort(df["energy"].unique().astype(float))
    Ax, Bx, Cx, Dx = _eval_coeffs_at_energies(fit_x, energies)
    Ay, By, Cy, Dy = _eval_coeffs_at_energies(fit_y, energies)
    x, xp, xxp = derived_params_at_zprime(Ax, Bx, Cx, Dx, z_prime)
    y, yp, yyp = derived_params_at_zprime(Ay, By, Cy, Dy, z_prime)
    out = pd.DataFrame({
        "energy": energies,
        "x_a": Ax, "x_b": Bx, "x_c": Cx, "x_d": Dx,
        "y_a": Ay, "y_b": By, "y_c": Cy, "y_d": Dy,
        "x_success": True,
        "y_success": True,
        "x": x, "y": y, "x'": xp,
        "y'": yp, "xx'": xxp, "yy'": yyp,
        "z": float(z_prime),
    })
    return out


def shift_reference(A, B, C, D, z_prime):
    A_prime = A
    B_prime = 2 * A * z_prime + B
    C_prime = A * z_prime**2 + B * z_prime + C + D * z_prime**3
    D_prime = D
    return A_prime, B_prime, C_prime, D_prime


def derived_params_at_zprime(A, B, C, D, z_prime):
    A_prime, B_prime, C_prime, D_prime = shift_reference(A, B, C, D, z_prime)
    x = np.sqrt(np.clip(C_prime, 0.0, None))
    xp = np.sqrt(np.clip(A_prime, 0.0, None))
    xxp = B_prime / 2.0
    return x, xp, xxp


def plot_fits(
    df: pd.DataFrame,
    fit_df: pd.DataFrame,
    output_prefix: str = "fit_plot",
    z0: float = 0.0
) -> None:
    fmap = {float(r["energy"]): r for _, r in fit_df.iterrows()}
    for energy in df["energy"].unique():
        energy = float(energy)
        g = df[df["energy"] == energy]
        z = g["z"].to_numpy(float)
        sx = g["sigma_x_mm"].to_numpy(float)
        sy = g["sigma_y_mm"].to_numpy(float)
        p = fmap[energy]
        z0_use = float(p.get("z0", z0))
        z_prime = p["z"]
        z_min = min(np.min(z), z_prime - 50)
        z_max = max(np.max(z), z_prime + 50)
        z_fit = np.linspace(z_min, z_max, 200)
        L = z_fit - z0_use
        x_fit = np.sqrt(np.clip(p["x_a"]*L**2 + p["x_b"]*L + p["x_c"] + p["x_d"]*L**3, 0.0, None))
        y_fit = np.sqrt(np.clip(p["y_a"]*L**2 + p["y_b"]*L + p["y_c"] + p["y_d"]*L**3, 0.0, None))
        plt.figure(figsize=(10, 6))
        plt.scatter(z, sx, label="x data")
        plt.scatter(z, sy, label="y data")
        plt.plot(z_fit, x_fit, "--", label="x fit")
        plt.plot(z_fit, y_fit, "--", label="y fit")
        plt.axvline(x=z0_use, color="gray", linestyle=":", label=f"Fit ref z0 = {z0_use:.1f} mm")
        plt.axvline(x=z_prime, color="red", linestyle=":", label=f"Params ref z' = {z_prime:.1f} mm")
        txt = (
            f"Energy = {energy:.1f} MeV\n"
            f"Fit ref z0 = {z0_use:.1f} mm\n"
            f"Params at z' = {z_prime:.1f} mm\n"
            f"Derived X: x={p['x']:.3f}, x'={p['x\'']:.3f}, xx'={p['xx\'']:.3f}\n"
            f"Derived Y: y={p['y']:.3f}, y'={p['y\'']:.3f}, yy'={p['yy\'']:.3f}"
        )
        plt.text(0.02, 0.95, txt, transform=plt.gca().transAxes,
                 va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        plt.xlabel("z (mm)")
        plt.ylabel("σ (mm)")
        plt.title(f"Energy = {energy:.1f} MeV")
        plt.grid(True)
        plt.legend()
        fname = f"{output_prefix}_energy_{energy:.1f}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved plot for energy %.1f MeV -> %s", energy, fname)
