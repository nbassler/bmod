import argparse
from pathlib import Path

from bmod.__version__ import __version__


def create_parser():
    parser = argparse.ArgumentParser(
        description="Analyse beam data")

    # get input directory
    parser.add_argument('input', type=Path,
                        help="Input file or directory.")

    parser.add_argument('-v', '--verbosity', action='count', default=0,
                        help="Increase verbosity (can use -v, -vv, etc.).")

    parser.add_argument('-V', '--version', action='version', version=__version__,
                        help="Show version and exit.")

    return parser
