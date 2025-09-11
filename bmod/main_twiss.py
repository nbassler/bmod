import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import argparse
import matplotlib.pyplot as plt
import warnings


def quadratic(z, a, b, c):
    """Simple quadratic function for fitting."""
    return a*z**2 + b*z + c


def fit_quadratic(group):
    """Fit quadratic to sigma^2 vs z for both x and y planes."""
    z = group['z'].to_numpy()
    sx = group['sigma_x_mm'].to_numpy()
    sy = group['sigma_y_mm'].to_numpy()

    # Fit x-plane
    try:
        popt_x, pcov_x = curve_fit(quadratic, z, sx**2, maxfev=10000)
        a_x, b_x, c_x = popt_x
        x_success = True
    except:
        a_x, b_x, c_x = np.nan, np.nan, np.nan
        x_success = False
        warnings.warn(f"Failed to fit x-plane at energy {group.name}")

    # Fit y-plane
    try:
        popt_y, pcov_y = curve_fit(quadratic, z, sy**2, maxfev=10000)
        a_y, b_y, c_y = popt_y
        y_success = True
    except:
        a_y, b_y, c_y = np.nan, np.nan, np.nan
        y_success = False
        warnings.warn(f"Failed to fit y-plane at energy {group.name}")

    return pd.Series({
        'energy': group.name,
        'x_a': a_x, 'x_b': b_x, 'x_c': c_x, 'x_success': x_success,
        'y_a': a_y, 'y_b': b_y, 'y_c': c_y, 'y_success': y_success
    })


def fit_all_energies(df):
    """Fit quadratic to sigma^2 vs z for each energy."""
    if not all(col in df.columns for col in ['z', 'sigma_x_mm', 'sigma_y_mm', 'energy']):
        raise ValueError("DataFrame must contain 'z', 'sigma_x_mm', 'sigma_y_mm', and 'energy' columns.")

    return df.groupby('energy').apply(fit_quadratic).reset_index(drop=True)


def plot_fits(df, fit_df):
    """Plot sigma^2 vs z with quadratic fits for each energy."""
    energies = df['energy'].unique()
    n_energies = len(energies)
    fig, axes = plt.subplots(n_energies, 1, figsize=(10, 4*n_energies))

    if n_energies == 1:
        axes = [axes]

    for i, energy in enumerate(energies):
        group = df[df['energy'] == energy]
        z = group['z'].to_numpy()
        sx = group['sigma_x_mm'].to_numpy()
        sy = group['sigma_y_mm'].to_numpy()

        params = fit_df[fit_df['energy'] == energy].iloc[0]
        ax = axes[i]

        # Plot data
        ax.scatter(z, sx**2, color='blue', label='x data')
        ax.scatter(z, sy**2, color='orange', label='y data')

        # Plot x fit if successful
        if params['x_success']:
            z_fit = np.linspace(min(z), max(z), 100)
            ax.plot(z_fit, quadratic(z_fit, params['x_a'], params['x_b'], params['x_c']),
                    '--', color='blue', label='x fit')

        # Plot y fit if successful
        if params['y_success']:
            z_fit = np.linspace(min(z), max(z), 100)
            ax.plot(z_fit, quadratic(z_fit, params['y_a'], params['y_b'], params['y_c']),
                    '--', color='orange', label='y fit')

        ax.set_xlabel('z (mm)')
        ax.set_ylabel('Sigma^2 (mm^2)')
        ax.set_title(f'Energy = {energy} MeV')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def main(input_file, output_file):
    """Main function to load data, fit quadratics, and plot results."""
    warnings.filterwarnings('always')

    # Load data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Data loaded. Found {len(df['energy'].unique())} unique energy levels.")

    # Fit quadratics
    print("Fitting quadratic functions...")
    fit_df = fit_all_energies(df)

    # Save results
    fit_df.to_csv(output_file, index=False)
    print(f"Fit parameters saved to {output_file}")

    # Print summary
    x_success = fit_df['x_success'].sum()
    y_success = fit_df['y_success'].sum()
    total = len(fit_df)
    print(f"Fit success rate: x-plane {x_success}/{total}, y-plane {y_success}/{total}")

    # Print parameters
    print("\nFitted quadratic parameters (σ² = a·z² + b·z + c):")
    print(fit_df[['energy', 'x_a', 'x_b', 'x_c', 'y_a', 'y_b', 'y_c']])

    # Plot results
    print("Generating plots...")
    plot_fits(df, fit_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit quadratic functions to beam size data.")
    parser.add_argument("input_file", type=str, help="Input CSV file (e.g., out.csv)")
    parser.add_argument("output_file", type=str, help="Output CSV file (e.g., quadratic_fits.csv)")
    args = parser.parse_args()

    main(args.input_file, args.output_file)
