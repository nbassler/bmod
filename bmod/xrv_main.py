# placeholder for geometry export functionality
import sys
import logging

import bmod.config_parser

from bmod.xrv_runner import run

logger = logging.getLogger(__name__)


def main(args=None) -> int:
    logger.debug("bmod.main called")

    # call parser:
    parser = bmod.config_parser.create_parser()
    parsed_args = parser.parse_args(args)

    logging.basicConfig(level=logging.WARNING)
    if parsed_args.verbosity == 1:
        logger.setLevel(logging.INFO)
    elif parsed_args.verbosity > 1:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    logger.debug("Parsed args: {:s}".format(str(parsed_args)))

    input_path = parsed_args.input

    # load configuration
    config = bmod.config_parser.load_config(parsed_args.config)
    logger.debug(f"Loaded config: {config}")

    if not input_path:
        logger.error("No input file or directory specified.")
        return 1

    run(input_path, config, write=parsed_args.output)

    return 0


if __name__ == '__main__':
    sys.exit(main())
