import argparse
import tomllib
from pathlib import Path

from bmod.__version__ import __version__


def create_parser():
    parser = argparse.ArgumentParser(
        description="Analyse beam data")

    # get input directory
    parser.add_argument('input', type=Path,
                        help="Input file or directory.")

    # add optional output path
    parser.add_argument('-o', '--output', type=Path, default=None,
                        help="Path to output CSV file or panda .parquet (optional).")

    # add path to toml configuration file (optional)
    parser.add_argument('-c', '--config', type=Path, default="bmod.toml",
                        help="Path to configuration file (TOML format).")

    parser.add_argument('-v', '--verbosity', action='count', default=0,
                        help="Increase verbosity (can use -v, -vv, etc.).")

    parser.add_argument('-V', '--version', action='version', version=__version__,
                        help="Show version and exit.")

    return parser


def load_config(config_path: Path) -> dict:
    """Load configuration from a TOML file."""
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
    return config
