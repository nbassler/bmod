import argparse
import logging
import sys
from zipfile import Path
import pandas as pd
# from xrv_twiss_quadratic import fit_all_energies as fit_quadratic, plot_fits as plot_quadratic
# from xrv_twiss_cubic import fit_all_energies as fit_cubic, plot_fits as plot_cubic
from bmod.xrv_twiss_quadratic_bspline import fit_all_energies as fit_quadratic, plot_fits as plot_quadratic
from bmod.xrv_twiss_cubic_bspline import fit_all_energies as fit_cubic, plot_fits as plot_cubic
from bmod.__version__ import __version__

# Get a logger for this module
logger = logging.getLogger(__name__)


def setup_parser():
    parser = argparse.ArgumentParser(description="Fit quadratic and cubic functions to beam size data.")
    parser.add_argument("input_file", type=str, help="Input CSV file (e.g., out.csv)")
    parser.add_argument("output_file", type=str, help="Output CSV file for combined results (e.g., combined_fits.csv)")
    # add optional input.csv file for subtracting air component
    parser.add_argument('-a', '--air_input', type=Path, default=None,
                        help="Path to input CSV file or panda .parquet (optional).")
    parser.add_argument('--z0', type=float, default=0.0,
                        help="Reference point for fitting, should be well surrounded by data points (default: 0.0 mm)")
    parser.add_argument('--z_beam_start', type=float, default=500.0,
                        help="Position for which the derived Twiss parameters will be prepared for (default: +500.0 mm)")
    parser.add_argument('--zdir_negative', action='store_true', default=True,
                        help="Beam travels in negative z direction. (default: True)")
    parser.add_argument('--no-plot', action='store_true', default=False,
                        help="Do not generate plots.")
    parser.add_argument('--cubic', action='store_true', default=False,
                        help="Also perform cubic fits.")
    parser.add_argument('-v', '--verbosity', action='count', default=0,
                        help="Increase verbosity (can use -v, -vv, etc.).")
    parser.add_argument('-V', '--version', action='version', version=__version__,
                        help="Show version and exit.")
    return parser


def subtract_air(df, df_air):
    """Subtract air data from measurement data based on matching energy levels
     and matching positions (z). Modifies df in place."""
    energies = df['energy'].unique()
    for energy in energies:
        air_subset = df_air[df_air['energy'] == energy]
        if air_subset.empty:
            logger.warning(f"No matching air data for energy {energy}. Skipping air subtraction for this energy.")
            continue
        for z in df['z'].unique():
            air_row = air_subset[air_subset['z'] == z]
            if air_row.empty:
                logger.warning(f"No matching air data for energy {energy} at z={z}. Skipping this position.")
                continue
            mask = (df['energy'].round().astype(int) == round(energy)) & (df['z'].round().astype(int) == round(z))
            df.loc[mask, 'sigma_x_mm'] = (df.loc[mask, 'sigma_x_mm']**2 -
                                          air_row['sigma_x_mm'].values[0]**2).clip(lower=0).pow(0.5)
            df.loc[mask, 'sigma_y_mm'] = (df.loc[mask, 'sigma_y_mm']**2 -
                                          air_row['sigma_y_mm'].values[0]**2).clip(lower=0).pow(0.5)


def main(args=None) -> int:
    """Main function: load data, fit quadratics and cubics, and plot results."""

    logging.basicConfig(level=logging.WARNING)  # must be setup before setting up the parser

    parser = setup_parser()
    args = parser.parse_args(args)

    input_file = args.input_file
    output_file = args.output_file
    air_file = args.air_input
    z0 = args.z0
    z_beam_start = args.z_beam_start
    zdir_negative = args.zdir_negative

    if input_file is None or output_file is None:
        parser.print_help()
        raise SystemExit(2)

    if args.verbosity == 1:
        logger.setLevel(logging.INFO)
    elif args.verbosity > 1:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    # Load data
    logger.info(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    logger.info(f"Data loaded. Found {len(df['energy'].unique())} unique energy levels.")

    if air_file is not None:
        logger.info(f"Loading air data from {air_file}...")
        if air_file.suffix == '.parquet':
            df_air = pd.read_parquet(air_file)
        else:
            df_air = pd.read_csv(air_file)
        logger.info(f"Air data loaded. Found {len(df_air['energy'].unique())} unique energy levels.")
        subtract_air(df, df_air)

    # Fit quadratics
    logger.info("Fitting quadratic functions...")
    quad_fit_df = fit_quadratic(df, z0=z0, z_prime=z_beam_start, zdir_negative=zdir_negative)
    # Save quadratic results
    quad_output_file = output_file.replace('.csv', '_quadratic.csv')
    quad_fit_df.to_csv(quad_output_file, index=False)
    logger.info(f"Quadratic fit parameters saved to {quad_output_file}")

    # Print quadratic summary
    x_success = quad_fit_df['x_success'].sum()
    y_success = quad_fit_df['y_success'].sum()
    total = len(quad_fit_df)
    logger.info(f"Quadratic fit success rate: x-plane {x_success}/{total}, y-plane {y_success}/{total}")
    logger.info("\nDerived beam parameters from quadratic fit:")
    logger.info(quad_fit_df[['energy', 'x', 'y', "x'", "y'", 'xx\'', 'yy\'']].to_string(index=False))

    # Plot quadratic fits
    if not args.no_plot:
        logger.info("Generating quadratic fit plots...")
        plot_quadratic(df, quad_fit_df, output_prefix="fit_plot_quadratic", z0=z0)

    if not args.cubic:
        return 0

    # the rest of the code is only executed if cubic fits are requested
    logger.info("Fitting cubic functions...")
    cubic_fit_df = fit_cubic(df, z0=z0)
    # Save cubic results
    cubic_output_file = output_file.replace('.csv', '_cubic.csv')
    cubic_fit_df.to_csv(cubic_output_file, index=False)
    logger.info(f"Cubic fit parameters saved to {cubic_output_file}")

    # Print cubic summary
    x_success = cubic_fit_df['x_success'].sum()
    y_success = cubic_fit_df['y_success'].sum()
    total = len(cubic_fit_df)

    logger.info(f"Cubic fit success rate: x-plane {x_success}/{total}, y-plane {y_success}/{total}")
    logger.info("\nFitted cubic parameters (σ² = a·(z-z0)² + b·(z-z0) + c + d·(z-z0)^3):")
    logger.info(cubic_fit_df[['energy', 'z0', 'x_a', 'x_b', 'x_c', 'x_d',
                              'y_a', 'y_b', 'y_c', 'y_d']].to_string(index=False))
    logger.info("\nDerived beam parameters from cubic fit:")
    logger.info(cubic_fit_df[['energy', 'x', 'y', "x'", "y'", 'xx\'', 'yy\'']].to_string(index=False))

    # Plot cubic fits
    logger.info("Generating cubic fit plots...")
    plot_cubic(df, cubic_fit_df, output_prefix="fit_plot_cubic", z0=z0)

    # Combine results into a single output file
    # quad_df = quad_fit_df.add_prefix('quad_')
    # cubic_df = cubic_fit_df.add_prefix('cubic_')
    # combined_df = pd.concat([quad_df, cubic_df], axis=1)
    # combined_df.to_csv(output_file, index=False)
    # logger.info(f"Combined fit parameters saved to {output_file}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
