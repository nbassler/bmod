import argparse
import logging
from xrv_twiss_quadratic import fit_all_energies, plot_fits
from __version__ import __version__

# Get a logger for this module
logger = logging.getLogger(__name__)


def main(input_file, output_file):
    """Main function: load data, fit quadratics, and plot results."""
    import pandas as pd  # Import here to avoid global namespace pollution

    # Load data
    logger.info(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    logger.info(f"Data loaded. Found {len(df['energy'].unique())} unique energy levels.")

    # Fit quadratics
    logger.info("Fitting quadratic functions...")
    fit_df = fit_all_energies(df)

    # Save results
    fit_df.to_csv(output_file, index=False)
    logger.info(f"Fit parameters saved to {output_file}")

    # Print summary
    x_success = fit_df['x_success'].sum()
    y_success = fit_df['y_success'].sum()
    total = len(fit_df)
    logger.info(f"Fit success rate: x-plane {x_success}/{total}, y-plane {y_success}/{total}")

    # Print parameters
    logger.info("\nFitted quadratic parameters (σ² = a·z² + b·z + c):")
    logger.info(fit_df[['energy', 'x_a', 'x_b', 'x_c', 'y_a', 'y_b', 'y_c']].to_string(index=False))

    # Plot results
    logger.info("Generating plots...")
    plot_fits(df, fit_df, output_prefix="fit_plot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit quadratic functions to beam size data.")
    parser.add_argument("input_file", type=str, help="Input CSV file (e.g., out.csv)")
    parser.add_argument("output_file", type=str, help="Output CSV file (e.g., quadratic_fits.csv)")
    parser.add_argument('-v', '--verbosity', action='count', default=0,
                        help="Increase verbosity (can use -v, -vv, etc.).")
    parser.add_argument('-V', '--version', action='version', version=__version__,
                        help="Show version and exit.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    if args.verbosity == 1:
        logger.setLevel(logging.INFO)
    elif args.verbosity > 1:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    main(args.input_file, args.output_file)
