"""Tests for beam-relative coordinate system and derived parameter evaluation."""
import numpy as np
import pandas as pd
import pytest
from bmod.xrv_twiss_quadratic_bspline import derived_params_at_zprime, fit_all_energies


SAD = 500.0  # typical source-axis distance in mm


def test_z_to_s_mapping():
    """Isocenter (IEC z=0) maps to s=SAD; beam start (IEC z=SAD) maps to s=0."""
    z = np.array([0.0, SAD, SAD / 2])
    s = SAD - z
    assert s[0] == pytest.approx(SAD)    # isocenter
    assert s[1] == pytest.approx(0.0)    # beam start
    assert s[2] == pytest.approx(SAD / 2)


def test_derived_params_at_beam_start():
    """derived_params_at_zprime evaluates sigma at the requested L-offset, not always at L=0.

    Polynomial: σ²(L) = A·L² + C  (B=0 for symmetry)
    At L=0 (fitting reference): σ = sqrt(C)
    At L=-s0 (beam start, s=0): σ = sqrt(A·s0² + C)
    These must differ when s0 != 0.
    """
    A, B, C = 1e-4, 0.0, 4.0  # simple parabola
    s0 = SAD  # fitting reference at isocenter

    # Evaluate at beam start: L = s_prime - s0 = 0 - SAD = -SAD
    L_beam_start = 0.0 - s0
    x_at_beam_start, _, _ = derived_params_at_zprime(A, B, C, L_beam_start)
    expected = np.sqrt(A * s0**2 + C)
    assert x_at_beam_start == pytest.approx(expected)

    # Sanity check: evaluating at L=0 gives sqrt(C), which is different
    x_at_ref, _, _ = derived_params_at_zprime(A, B, C, 0.0)
    assert x_at_ref == pytest.approx(np.sqrt(C))
    assert x_at_beam_start != pytest.approx(x_at_ref)


def _make_synthetic_df(sad=SAD, n_z=10, energies=(100.0, 200.0)):
    """Create a minimal synthetic DataFrame with s-coordinates and known parabolic sigma."""
    z_vals = np.linspace(-100, 100, n_z)  # IEC z around isocenter
    rows = []
    for e in energies:
        for z in z_vals:
            s = sad - z
            L = s - sad  # s0=sad (IEC z0=0)
            sigma = np.sqrt(max(0.01 * L**2 + 4.0, 0.0))
            rows.append({"z": z, "s": s, "energy": e,
                         "sigma_x_mm": sigma, "sigma_y_mm": sigma * 0.9})
    return pd.DataFrame(rows)


def test_fit_all_energies_s_column_required():
    """fit_all_energies raises ValueError if 's' column is missing."""
    df = _make_synthetic_df()
    df_no_s = df.drop(columns=["s"])
    with pytest.raises(ValueError, match="'s'"):
        fit_all_energies(df_no_s)


def test_fit_all_energies_derived_params_at_beam_start():
    """Derived sigma returned by fit_all_energies for s_prime=0 must be > 0 and physically reasonable."""
    df = _make_synthetic_df()
    s0 = SAD  # fitting reference at isocenter (IEC z0=0)
    result = fit_all_energies(df, s0=s0, s_prime=0.0)

    assert len(result) == 2  # one row per energy
    assert (result["s"] == 0.0).all()  # params recorded at s=0
    # sigma at beam start (s=0, L=-SAD) should be larger than at isocenter (L=0)
    # because the beam diverges from the source
    assert (result["x"] > 0).all()
    assert (result["y"] > 0).all()
