import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import logging

logger = logging.getLogger(__name__)


def cubic(z, a, b, c, d, z0=-500.0):
    """Cubic function for fitting: σ² = a·(z-z0)² + b·(z-z0) + c + d·(z-z0)^3."""
    L = z - z0
    return a * L**2 + b * L + c + d * L**3


def fit_cubic(group, z0=-500.0):
    """Fit cubic to σ² vs z for both x and y planes."""
    z = group['z'].to_numpy()
    sx = group['sigma_x_mm'].to_numpy()
    sy = group['sigma_y_mm'].to_numpy()

    # Define a version of cubic with z0 fixed
    def cubic_fixed_z0(z, a, b, c, d):
        return cubic(z, a, b, c, d, z0)

    # Fit x-plane
    try:
        # Initial guesses - a, b, c similar to quadratic, d (for L³ term) starts small
        initial_guess_x = [1e-3, 1e-3, 1.0, 1e-6]
        popt_x, pcov_x = curve_fit(cubic_fixed_z0, z, sx**2, p0=initial_guess_x, maxfev=10000)
        a_x, b_x, c_x, d_x = popt_x
        x_success = True
    except Exception as e:
        print(f"Failed to fit x-plane at energy {group.name}: {e}")
        a_x, b_x, c_x, d_x = np.nan, np.nan, np.nan, np.nan
        x_success = False
        logger.warning(f"Failed to fit x-plane at energy {group.name}")

    # Fit y-plane
    try:
        initial_guess_y = [1e-3, 1e-3, 1.0, 1e-6]
        popt_y, pcov_y = curve_fit(cubic_fixed_z0, z, sy**2, p0=initial_guess_y, maxfev=10000)
        a_y, b_y, c_y, d_y = popt_y
        y_success = True
    except Exception as e:
        print(f"Failed to fit y-plane at energy {group.name}: {e}")
        a_y, b_y, c_y, d_y = np.nan, np.nan, np.nan, np.nan
        y_success = False
        logger.warning(f"Failed to fit y-plane at energy {group.name}")

    # Ensure c_x and c_y are non-negative (since we take sqrt for beam size)
    c_x = abs(c_x) if not np.isnan(c_x) else np.nan
    c_y = abs(c_y) if not np.isnan(c_y) else np.nan

    # Calculate derived parameters
    x = np.sqrt(c_x) if c_x >= 0 else np.nan  # beam size
    y = np.sqrt(c_y) if c_y >= 0 else np.nan
    xp = np.sqrt(abs(a_x)) if not np.isnan(a_x) else np.nan  # divergence (abs to avoid complex)
    yp = np.sqrt(abs(a_y)) if not np.isnan(a_y) else np.nan
    xxp = b_x / 2 if not np.isnan(b_x) else np.nan  # correlation
    yyp = b_y / 2 if not np.isnan(b_y) else np.nan

    return pd.Series({
        'energy': group.name,
        'x_a': a_x, 'x_b': b_x, 'x_c': c_x, 'x_d': d_x, 'x_success': x_success,
        'y_a': a_y, 'y_b': b_y, 'y_c': c_y, 'y_d': d_y, 'y_success': y_success,
        'x': x, 'y': y,
        'x\'': xp, 'y\'': yp,
        'xx\'': xxp, 'yy\'': yyp,
        'z0': z0  # include z0 in the output
    })


def fit_all_energies(df, z0=-500.0):
    """Fit cubic to σ² vs z for each energy."""
    required_columns = ['z', 'sigma_x_mm', 'sigma_y_mm', 'energy']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    return df.groupby('energy').apply(fit_cubic, z0=z0).reset_index(drop=True)


def plot_fits(df, fit_df, output_prefix="fit_plot", z0=-500.0):
    """Plot σ² vs z with cubic fits for each energy."""
    import matplotlib.pyplot as plt

    for energy in df['energy'].unique():
        plt.figure(figsize=(10, 6))
        group = df[df['energy'] == energy]
        z = group['z'].to_numpy()
        sx = group['sigma_x_mm'].to_numpy()
        sy = group['sigma_y_mm'].to_numpy()
        params = fit_df[fit_df['energy'] == energy].iloc[0]

        # Use z0 from params if it's there, otherwise use the default
        z0_to_use = params.get('z0', z0)

        # Plot data
        plt.scatter(z, sx**2, color='blue', label='x data')
        plt.scatter(z, sy**2, color='orange', label='y data')

        # Plot fits if successful
        if params['x_success']:
            z_fit = np.linspace(min(z), max(z), 100)
            plt.plot(z_fit, cubic(z_fit, params['x_a'], params['x_b'], params['x_c'], params['x_d'], z0_to_use),
                     '--', color='blue', label='x fit')
        if params['y_success']:
            z_fit = np.linspace(min(z), max(z), 100)
            plt.plot(z_fit, cubic(z_fit, params['y_a'], params['y_b'], params['y_c'], params['y_d'], z0_to_use),
                     '--', color='orange', label='y fit')

        # Add fit parameters as text
        fit_info = (
            f"Energy = {energy} MeV\n"
            f"z0 = {z0_to_use} mm\n"
            f"X: a={params['x_a']:.2e}, b={params['x_b']:.2e}, c={params['x_c']:.2e}, d={params['x_d']:.2e}\n"
            f"Y: a={params['y_a']:.2e}, b={params['y_b']:.2e}, c={params['y_c']:.2e}, d={params['y_d']:.2e}"
        )
        plt.text(0.02, 0.95, fit_info, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.xlabel('z (mm)')
        plt.ylabel('σ² (mm²)')
        plt.title(f'Energy = {energy} MeV')
        plt.legend()
        plt.grid(True)
        # Save the figure
        filename = f"{output_prefix}_energy_{energy:.1f}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot for energy {energy} MeV to {filename}")
