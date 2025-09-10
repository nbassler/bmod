# placeholder for geometry export functionality
import sys
import logging

import bmod.config_parser

logger = logging.getLogger(__name__)


def main(args=None) -> int:
    logger.debug("bmod.main called")

    # call parser:
    parser = bmod.config_parser.create_parser()
    parsed_args = parser.parse_args(args)

    # Set up logging based on verbosity
    if parsed_args.verbosity == 1:
        logging.basicConfig(level=logging.INFO)
    elif parsed_args.verbosity > 1:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig()

    logger.debug("Parsed args: {:s}".format(str(parsed_args)))

    input_path = parsed_args.input

    if not input_path:
        logger.error("No input file or directory specified.")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
