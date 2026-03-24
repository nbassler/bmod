# xrv_energyfit_quadratic.py
from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# ----------------------------
# Helpers: normalization, basis, penalty
# ----------------------------


def _normalize(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    scale = (xmax - xmin) if xmax > xmin else 1.0
    return (x - xmin) / scale, xmin, scale


def _make_bspline_basis(x: np.ndarray, degree: int = 3, n_bases: int = 8) -> tuple[np.ndarray, np.ndarray]:
    if n_bases < degree + 1:
        n_bases = degree + 1
    m = n_bases - degree - 1  # number of internal knots
    internal = np.linspace(0, 1, m + 2)[1:-1] if m > 0 else np.array([])
    t = np.concatenate([np.zeros(degree + 1), internal, np.ones(degree + 1)])
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
# Global quadratic-in-L model per plane
# σ^2(z,E) = A(E) L^2 + B(E) L + C(E),  L = z - z0
# A,B,C are cubic B-splines in E
# ----------------------------


class _PlaneFit:
    __slots__ = ("coef_A", "coef_B", "coef_C", "knots", "degree", "E_min", "E_scale", "n_bases")

    def __init__(self, coef_A, coef_B, coef_C, knots, degree, E_min, E_scale, n_bases):
        self.coef_A = coef_A
        self.coef_B = coef_B
        self.coef_C = coef_C
        self.knots = knots
        self.degree = degree
        self.E_min = E_min
        self.E_scale = E_scale
        self.n_bases = n_bases


def _fit_plane_global(
    df: pd.DataFrame, s0: float, plane: str,
    *, n_bases: int = 10, degree: int = 3,
    lambda_reg: tuple[float, float, float] = (1e-2, 1e-2, 1e-2),
    weight_col: Optional[str] = None
) -> _PlaneFit:
    """Fit one plane (x or y) with global B-spline in energy.
    n_bases: number of B-spline bases
    degree: degree of B-spline (typically 3)
    lambda_reg: regularization strength for [A,B,C] blocks
    weight_col: optional column name for weights (0..1), default equal weights"""

    s = df["s"].to_numpy(float)
    E = df["energy"].to_numpy(float)
    sig = df[f"sigma_{plane}_mm"].to_numpy(float)

    E_n, E_min, E_scale = _normalize(E)
    B, knots = _make_bspline_basis(E_n, degree=degree, n_bases=n_bases)  # N x K
    K = B.shape[1]
    L = s - float(s0)

    # Design matrix: [L^2*B | L*B | 1*B]   -> N x (3K)
    X = np.concatenate([(L**2)[:, None]*B, L[:, None]*B, B], axis=1)
    y = np.square(sig)

    # weights
    if weight_col and (weight_col in df.columns):
        w = np.sqrt(np.clip(df[weight_col].to_numpy(float), 1e-6, 1.0))
    else:
        w = np.ones_like(y)

    WX = w[:, None] * X
    Wy = w * y

    # smoothness (second-diff) per block A,B,C
    D2 = _second_diff_matrix(K)
    rows = []
    for i, lam in enumerate(lambda_reg):
        if lam <= 0 or D2.size == 0:
            continue
        row = np.zeros((D2.shape[0], 3*K))
        row[:, i*K:(i+1)*K] = np.sqrt(lam) * D2
        rows.append(row)
    R = np.vstack(rows) if rows else np.zeros((0, 3*K))

    A_mat = np.vstack([WX, R])
    b_vec = np.concatenate([Wy, np.zeros(R.shape[0])])

    theta, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)  # (3K,)
    coef_A = theta[0*K:1*K]
    coef_B = theta[1*K:2*K]
    coef_C = theta[2*K:3*K]
    return _PlaneFit(coef_A, coef_B, coef_C, knots, degree, E_min, E_scale, K)


def _eval_coeffs(fit: _PlaneFit, energies: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    E_n = (energies - fit.E_min) / (fit.E_scale if fit.E_scale != 0 else 1.0)
    B = np.empty((E_n.size, fit.n_bases), dtype=float)
    for j in range(fit.n_bases):
        c = np.zeros(fit.n_bases)
        c[j] = 1.0
        B[:, j] = BSpline(fit.knots, c, fit.degree)(E_n)
    A = B @ fit.coef_A
    Bc = B @ fit.coef_B
    C = B @ fit.coef_C
    return A, Bc, C

# ----------------------------
# Public API (unchanged)
# ----------------------------


def fit_all_energies(
    df: pd.DataFrame,
    s0: float = 0.0,
    s_prime: float = 0.0,
    *, n_bases: int = 8, degree: int = 3,
    lambda_reg_xy: tuple[float, float, float] = (1e-2, 1e-2, 1e-2),
    weight_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Global smooth fit across energy for quadratic-in-L model.
    Returns per-energy coefficients and derived params.
    Required df columns: ['s','sigma_x_mm','sigma_y_mm','energy']
    The 's' column is the beam-direction coordinate (s=0 at beam start, s=SAD at isocenter).
    s0: reference point for fitting (in beam coordinates)
    s_prime: position where derived beam parameters are evaluated (in beam coordinates)
    """
    req = ['s', 'sigma_x_mm', 'sigma_y_mm', 'energy']
    if not all(c in df.columns for c in req):
        raise ValueError(f"DataFrame must contain columns: {req}")

    fit_x = _fit_plane_global(df, s0, "x", n_bases=n_bases, degree=degree,
                              lambda_reg=lambda_reg_xy, weight_col=weight_col)
    fit_y = _fit_plane_global(df, s0, "y", n_bases=n_bases, degree=degree,
                              lambda_reg=lambda_reg_xy, weight_col=weight_col)

    energies = np.sort(df["energy"].unique().astype(float))
    Ax, Bx, Cx = _eval_coeffs(fit_x, energies)
    Ay, By, Cy = _eval_coeffs(fit_y, energies)

    # σ^2(s,E) = A(E) L^2 + B(E) L + C(E),  L = s - s0
    # derived_params_at_zprime expects an L-offset, so convert absolute s_prime
    L_prime = s_prime - s0
    x, xp, xxp = derived_params_at_zprime(Ax, Bx, Cx, L_prime)
    y, yp, yyp = derived_params_at_zprime(Ay, By, Cy, L_prime)

    out = pd.DataFrame({
        "energy": energies,
        "x_a": Ax, "x_b": Bx, "x_c": Cx,
        "y_a": Ay, "y_b": By, "y_c": Cy,
        "x_success": True,
        "y_success": True,
        "x": x, "y": y, "x'": xp,
        "y'": yp, "xx'": xxp, "yy'": yyp,
        "s": float(s_prime),
    })
    return out


def shift_reference(A, B, C, s_prime):
    A_prime = A
    B_prime = 2 * A * s_prime + B
    C_prime = A * s_prime**2 + B * s_prime + C
    return A_prime, B_prime, C_prime


def derived_params_at_zprime(A, B, C, s_prime):
    A_prime, B_prime, C_prime = shift_reference(A, B, C, s_prime)
    x = np.sqrt(np.clip(C_prime, 0.0, None))
    xp = np.sqrt(np.clip(A_prime, 0.0, None))
    xxp = B_prime / 2.0
    return x, xp, xxp


def plot_fits(
    df: pd.DataFrame, fit_df: pd.DataFrame,
    output_prefix: str = "fit_plot", s0: float = 0.0
) -> None:
    """Plot σ vs s and the quadratic fits per energy, showing derived x and y parameters at s'."""
    fmap = {float(r["energy"]): r for _, r in fit_df.iterrows()}
    for energy in df["energy"].unique():
        energy = float(energy)
        g = df[df["energy"] == energy]
        s = g["s"].to_numpy(float)
        sx = g["sigma_x_mm"].to_numpy(float)
        sy = g["sigma_y_mm"].to_numpy(float)
        p = fmap[energy]
        s0_use = float(p.get("s0", s0))  # Original fit reference
        s_prime = p["s"]  # Derived parameter reference

        # Extend s_fit range to include s_prime
        s_min = min(np.min(s), s_prime - 50)
        s_max = max(np.max(s), s_prime + 50)
        s_fit = np.linspace(s_min, s_max, 200)

        L = s_fit - s0_use
        # Calculate sigma from sigma² fits
        x_fit = np.sqrt(np.clip(p["x_a"]*L**2 + p["x_b"]*L + p["x_c"], 0.0, None))
        y_fit = np.sqrt(np.clip(p["y_a"]*L**2 + p["y_b"]*L + p["y_c"], 0.0, None))

        plt.figure(figsize=(10, 6))
        plt.scatter(s, sx, label="x data")
        plt.scatter(s, sy, label="y data")
        plt.plot(s_fit, x_fit, "--", label="x fit")
        plt.plot(s_fit, y_fit, "--", label="y fit")

        plt.axvline(x=s0_use, color="gray", linestyle=":", label=f"s_ref = {s0_use:.1f} mm")
        plt.axvline(x=s_prime, color="red", linestyle=":", label=f"s = {s_prime:.1f} mm")

        xp, xxp, yp, yyp = p["x'"], p["xx'"], p["y'"], p["yy'"]
        txt = (
            f"Energy = {energy:.1f} MeV\n"
            f"s_ref = {s0_use:.1f} mm\n"
            f"Params at s = {s_prime:.1f} mm\n"
            f"Derived X: x={p['x']:.3f}, x'={xp:.3f}, xx'={xxp:.3f}\n"
            f"Derived Y: y={p['y']:.3f}, y'={yp:.3f}, yy'={yyp:.3f}"
        )
        plt.text(0.02, 0.95, txt, transform=plt.gca().transAxes,
                 va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        plt.xlabel("s (mm)")
        plt.ylabel("σ (mm)")
        plt.title(f"Energy = {energy:.1f} MeV")
        plt.grid(True)
        plt.legend()
        fname = f"{output_prefix}_energy_{energy:.1f}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved plot for energy %.1f MeV -> %s", energy, fname)
