from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

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
    """
    Build a cubic B-spline design matrix for x in [0, 1].
    Returns (B, t) where:
      - B is [N x n_bases] with basis evaluations
      - t is the knot vector used (for reference)
    """
    if n_bases < degree + 1:
        n_bases = degree + 1

    # Internal knots count m = n_bases - degree - 1
    m = n_bases - degree - 1
    if m > 0:
        # uniform internal knots in (0,1)
        internal = np.linspace(0, 1, m + 2)[1:-1]
    else:
        internal = np.array([])

    # full knot vector with boundary multiplicity degree+1
    t = np.concatenate([
        np.zeros(degree + 1),
        internal,
        np.ones(degree + 1)
    ])

    # Evaluate each basis by using identity coefficients
    B = np.empty((x.size, n_bases), dtype=float)
    for j in range(n_bases):
        c = np.zeros(n_bases)
        c[j] = 1.0
        B[:, j] = BSpline(t, c, degree)(x)
    return B, t


def _second_diff_matrix(n: int) -> np.ndarray:
    """Second-difference operator matrix (n-2 x n) for smoothing penalty."""
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
    # Spline coefficient vectors for A(E), B(E), C(E), D(E)
    coef_A: np.ndarray
    coef_B: np.ndarray
    coef_C: np.ndarray
    coef_D: np.ndarray
    knots:   np.ndarray
    degree:  int
    # Normalization used for E
    E_min: float
    E_scale: float
    # Basis count
    n_bases: int


def _fit_plane_global(
    df: pd.DataFrame,
    z0: float,
    plane: str,                          # "x" or "y"
    *,
    n_bases: int = 8,
    degree: int = 3,
    lambda_reg: Tuple[float, float, float, float] = (1e-2, 1e-2, 1e-2, 1e-2),
    weight_col: Optional[str] = None     # e.g. "r2" to weight by fit quality
) -> _PlaneFit:
    """
    Fit one global model for a plane (x or y):
      sigma_plane^2(z,E) = A(E)L^2 + B(E)L + C(E) + D(E)L^3,  L = z - z0
    where A..D are cubic B-splines in E.
    """

    # Pull vectors
    z = df["z"].to_numpy(dtype=float)
    E = df["energy"].to_numpy(dtype=float)
    sigma = df[f"sigma_{plane}_mm"].to_numpy(dtype=float)

    # Normalize energy to [0,1] for basis stability
    E_n, E_min, E_scale = _normalize(E)

    # Build spline basis B(E) -> [N x n_bases]
    B, knots = _make_bspline_basis(E_n, degree=degree, n_bases=n_bases)  # N x K
    K = B.shape[1]
    N = B.shape[0]

    # Build z polynomial columns
    L = z - float(z0)
    Z2 = L**2
    Z1 = L
    Z0 = np.ones_like(L)
    Z3 = L**3

    # Design matrix X = [Z2*B | Z1*B | Z0*B | Z3*B]  -> N x (4K)
    X_blocks = [Z2[:, None] * B, Z1[:, None] * B, Z0[:, None] * B, Z3[:, None] * B]
    X = np.concatenate(X_blocks, axis=1)  # N x (4K)

    # Target: y = sigma^2
    y = np.square(sigma)

    # Row weights
    if weight_col is not None and weight_col in df.columns:
        w_raw = np.asarray(df[weight_col].to_numpy(dtype=float))
        w_raw = np.clip(w_raw, 1e-6, 1.0)  # keep finite
    else:
        w_raw = np.ones_like(y)

    # Scale to stabilize
    w = np.sqrt(w_raw)

    # Tikhonov regularization on second diffs within each block
    D2 = _second_diff_matrix(K)          # (K-2) x K
    blocks = []
    for lam in lambda_reg:
        if D2.size == 0 or lam <= 0:
            continue
        blocks.append(np.sqrt(lam) * np.pad(D2, ((0, 0), (0, 0))))  # each block K columns
    # Build block-diagonal for 4 blocks
    if D2.size == 0:
        R = np.zeros((0, 4 * K))
    else:
        # stack block-diagonal by placing D2 in the right column ranges
        rows = []
        for i, lam in enumerate(lambda_reg):
            if lam <= 0:
                continue
            row = np.zeros((D2.shape[0], 4 * K))
            col0 = i * K
            row[:, col0:col0 + K] = np.sqrt(lam) * D2
            rows.append(row)
        R = np.vstack(rows) if rows else np.zeros((0, 4 * K))

    # Weighted normal equations with regularization:
    #   [W X; R] theta ≈ [W y; 0]
    WX = (w[:, None] * X)
    Wy = (w * y)
    A = np.vstack([WX, R])
    b = np.concatenate([Wy, np.zeros(R.shape[0])])

    # Solve least squares
    theta, *_ = np.linalg.lstsq(A, b, rcond=None)  # (4K,)

    # Split theta into 4 blocks
    coef_A = theta[0*K:1*K]
    coef_B = theta[1*K:2*K]
    coef_C = theta[2*K:3*K]
    coef_D = theta[3*K:4*K]

    return _PlaneFit(
        coef_A=coef_A, coef_B=coef_B, coef_C=coef_C, coef_D=coef_D,
        knots=knots, degree=degree, E_min=E_min, E_scale=E_scale, n_bases=K
    )


def _eval_coeffs_at_energies(fit: _PlaneFit, energies: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate A(E), B(E), C(E), D(E) at desired energy points."""
    E_n = (energies - fit.E_min) / (fit.E_scale if fit.E_scale != 0 else 1.0)
    # build basis at E_n using the same knots/degree
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
# Public API (unchanged names)
# ----------------------------

def fit_all_energies(
    df: pd.DataFrame,
    z0: float = -500.0,
    *,
    n_bases: int = 8,
    degree: int = 3,
    lambda_reg_xy: Tuple[float, float, float, float] = (1e-2, 1e-2, 1e-2, 1e-2),
    weight_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Global semi-parametric fit across energies.
    Returns a DataFrame with per-energy coefficients like before:
      energy, x_a, x_b, x_c, x_d, y_a, y_b, y_c, y_d, x, y, x', y', xx', yy', z0
    Required columns in df: ['z', 'sigma_x_mm', 'sigma_y_mm', 'energy']
    """
    required = ['z', 'sigma_x_mm', 'sigma_y_mm', 'energy']
    if not all(c in df.columns for c in required):
        raise ValueError(f"DataFrame must contain columns: {required}")

    # Fit both planes globally
    fit_x = _fit_plane_global(df, z0, plane="x",
                              n_bases=n_bases, degree=degree,
                              lambda_reg=lambda_reg_xy, weight_col=weight_col)
    fit_y = _fit_plane_global(df, z0, plane="y",
                              n_bases=n_bases, degree=degree,
                              lambda_reg=lambda_reg_xy, weight_col=weight_col)

    # Unique energies, sorted, to report per-energy coefficients
    energies = np.sort(df["energy"].unique().astype(float))

    Ax, Bx, Cx, Dx = _eval_coeffs_at_energies(fit_x, energies)
    Ay, By, Cy, Dy = _eval_coeffs_at_energies(fit_y, energies)

    # Derived "Twiss-like" summaries at z=L=0:
    # sigma^2(L=0) = C  -> sigma = sqrt(max(C,0))
    # A is coeff of L^2; divergence proxy sqrt(max(A,0))
    # B relates to correlation; xx' ~ B/2
    x = np.sqrt(np.clip(Cx, 0.0, None))
    y = np.sqrt(np.clip(Cy, 0.0, None))
    xp = np.sqrt(np.clip(Ax, 0.0, None))
    yp = np.sqrt(np.clip(Ay, 0.0, None))
    xxp = Bx / 2.0
    yyp = By / 2.0

    out = pd.DataFrame({
        "energy": energies,
        "x_a": Ax, "x_b": Bx, "x_c": Cx, "x_d": Dx,
        "y_a": Ay, "y_b": By, "y_c": Cy, "y_d": Dy,
        "x": x, "y": y, "x'": xp, "y'": yp, "xx'": xxp, "yy'": yyp,
        "z0": z0,
        # flags for compatibility
        "x_success": True,
        "y_success": True,
    })

    return out


def plot_fits(
    df: pd.DataFrame,
    fit_df: pd.DataFrame,
    output_prefix: str = "fit_plot",
    z0: float = -500.0
) -> None:
    """
    Plot σ² vs z with global cubic-in-L fits for each energy.
    Assumes `fit_df` came from fit_all_energies (same API as before).
    """
    # Build a quick lookup from energy -> row of fit_df
    fit_map = {float(row["energy"]): row for _, row in fit_df.iterrows()}

    for energy in df["energy"].unique():
        energy = float(energy)
        group = df[df["energy"] == energy]
        z = group["z"].to_numpy(dtype=float)
        sx = group["sigma_x_mm"].to_numpy(dtype=float)
        sy = group["sigma_y_mm"].to_numpy(dtype=float)

        params = fit_map[energy]
        # Use z0 from params if present
        z0_to_use = float(params.get("z0", z0))

        # Plot data
        plt.figure(figsize=(10, 6))
        plt.scatter(z, sx**2, label="x data")
        plt.scatter(z, sy**2, label="y data")

        # Smooth curves from the per-energy coefficients
        z_fit = np.linspace(np.min(z), np.max(z), 200)
        L = z_fit - z0_to_use
        x_fit = (params["x_a"] * L**2 + params["x_b"] * L + params["x_c"] + params["x_d"] * L**3)
        y_fit = (params["y_a"] * L**2 + params["y_b"] * L + params["y_c"] + params["y_d"] * L**3)

        plt.plot(z_fit, x_fit, "--", label="x fit")
        plt.plot(z_fit, y_fit, "--", label="y fit")

        # annotate
        fit_info = (
            f"Energy = {energy} MeV\n"
            f"z0 = {z0_to_use:.1f} mm\n"
            f"X: a={params['x_a']:.2e}, b={params['x_b']:.2e}, c={params['x_c']:.2e}, d={params['x_d']:.2e}\n"
            f"Y: a={params['y_a']:.2e}, b={params['y_b']:.2e}, c={params['y_c']:.2e}, d={params['y_d']:.2e}"
        )
        plt.text(0.02, 0.95, fit_info, transform=plt.gca().transAxes,
                 va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        plt.xlabel("z (mm)")
        plt.ylabel("σ² (mm²)")
        plt.title(f"Energy = {energy:.1f} MeV")
        plt.legend()
        plt.grid(True)

        fname = f"{output_prefix}_energy_{energy:.1f}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved plot for energy %.1f MeV -> %s", energy, fname)
