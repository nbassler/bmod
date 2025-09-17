import argparse
import logging
import sys
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
    parser.add_argument('--z0', type=float, default=0.0,
                        help="Offset for z coordinate (default: 0.0 mm)")
    parser.add_argument('--no-plot', action='store_true', default=False,
                        help="Do not generate plots.")
    parser.add_argument('--cubic', action='store_true', default=False,
                        help="Also perform cubic fits.")
    parser.add_argument('-v', '--verbosity', action='count', default=0,
                        help="Increase verbosity (can use -v, -vv, etc.).")
    parser.add_argument('-V', '--version', action='version', version=__version__,
                        help="Show version and exit.")
    return parser


def main(args=None) -> int:
    """Main function: load data, fit quadratics and cubics, and plot results."""

    logging.basicConfig(level=logging.WARNING)  # must be setup before setting up the parser
    parser = setup_parser()
    args = parser.parse_args(args)
    input_file = args.input_file
    output_file = args.output_file

    if input_file is None or output_file is None:
        parser.print_help()
        raise SystemExit(2)

    z0 = args.z0

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

    # Fit quadratics
    logger.info("Fitting quadratic functions...")
    quad_fit_df = fit_quadratic(df, z0=z0)
    # Save quadratic results
    quad_output_file = output_file.replace('.csv', '_quadratic.csv')
    quad_fit_df.to_csv(quad_output_file, index=False)
    logger.info(f"Quadratic fit parameters saved to {quad_output_file}")

    # Print quadratic summary
    x_success = quad_fit_df['x_success'].sum()
    y_success = quad_fit_df['y_success'].sum()
    total = len(quad_fit_df)
    logger.info(f"Quadratic fit success rate: x-plane {x_success}/{total}, y-plane {y_success}/{total}")
    logger.info("\nFitted quadratic parameters (σ² = a·(z-z0)² + b·(z-z0) + c):")
    logger.info(quad_fit_df[['energy', 'z0', 'x_a', 'x_b', 'x_c', 'y_a', 'y_b', 'y_c']].to_string(index=False))
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
