import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import logging

logger = logging.getLogger(__name__)


def quadratic(z, a, b, c):
    """Quadratic function for fitting: σ² = a·z² + b·z + c."""
    return a * z**2 + b * z + c


def fit_quadratic(group):
    """Fit quadratic to σ² vs z for both x and y planes."""
    z = group['z'].to_numpy()
    sx = group['sigma_x_mm'].to_numpy()
    sy = group['sigma_y_mm'].to_numpy()

    # Fit x-plane
    try:
        popt_x, pcov_x = curve_fit(quadratic, z, sx**2, maxfev=10000)
        a_x, b_x, c_x = popt_x
        x_success = True
    except Exception as e:
        print(f"Failed to fit x-plane at energy {group.name}: {e}")
        a_x, b_x, c_x = np.nan, np.nan, np.nan
        x_success = False
        logger.warning(f"Failed to fit x-plane at energy {group.name}")

    # Fit y-plane
    try:
        popt_y, pcov_y = curve_fit(quadratic, z, sy**2, maxfev=10000)
        a_y, b_y, c_y = popt_y
        y_success = True
    except Exception as e:
        print(f"Failed to fit y-plane at energy {group.name}: {e}")
        a_y, b_y, c_y = np.nan, np.nan, np.nan
        y_success = False
        logger.warning(f"Failed to fit y-plane at energy {group.name}")

    x = np.sqrt(c_x)  # beam size
    y = np.sqrt(c_y)
    xp = np.sqrt(a_x)  # divergence
    yp = np.sqrt(a_y)
    xxp = b_x / 2  # correlation
    yyp = b_y / 2

    return pd.Series({
        'energy': group.name,
        'x_a': a_x, 'x_b': b_x, 'x_c': c_x, 'x_success': x_success,
        'y_a': a_y, 'y_b': b_y, 'y_c': c_y, 'y_success': y_success,
        'x': x, 'y': y,
        'x\'': xp, 'y\'': yp,
        'xx\'': xxp, 'yy\'': yyp
    })


def fit_all_energies(df):
    """Fit quadratic to σ² vs z for each energy."""
    required_columns = ['z', 'sigma_x_mm', 'sigma_y_mm', 'energy']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    return df.groupby('energy').apply(fit_quadratic).reset_index(drop=True)


def plot_fits(df, fit_df, output_prefix="fit_plot"):
    """Plot σ² vs z with quadratic fits for each energy."""
    import matplotlib.pyplot as plt  # Import here to avoid requiring plt for non-plotting use

    for energy in df['energy'].unique():
        plt.figure(figsize=(10, 6))
        group = df[df['energy'] == energy]
        z = group['z'].to_numpy()
        sx = group['sigma_x_mm'].to_numpy()
        sy = group['sigma_y_mm'].to_numpy()
        params = fit_df[fit_df['energy'] == energy].iloc[0]

        # Plot data
        plt.scatter(z, sx**2, color='blue', label='x data')
        plt.scatter(z, sy**2, color='orange', label='y data')

        # Plot fits if successful
        if params['x_success']:
            z_fit = np.linspace(min(z), max(z), 100)
            plt.plot(z_fit, quadratic(z_fit, params['x_a'], params['x_b'], params['x_c']),
                     '--', color='blue', label='x fit')
        if params['y_success']:
            z_fit = np.linspace(min(z), max(z), 100)
            plt.plot(z_fit, quadratic(z_fit, params['y_a'], params['y_b'], params['y_c']),
                     '--', color='orange', label='y fit')

        # Add fit parameters as text
        fit_info = (
            f"Energy = {energy} MeV\n"
            f"X: a={params['x_a']:.2e}, b={params['x_b']:.2e}, c={params['x_c']:.2e}\n"
            f"Y: a={params['y_a']:.2e}, b={params['y_b']:.2e}, c={params['y_c']:.2e}"
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
