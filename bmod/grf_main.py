# placeholder for geometry export functionality
import sys
import logging

import bmod.config_parser

from bmod.grf_runner import run


def main(args=None) -> int:

    # setup root logger:
    logging.basicConfig(level=logging.WARNING)

    # setup module logger:
    logger = logging.getLogger("bmod")

    # call parser:
    parser = bmod.config_parser.create_parser()
    parsed_args = parser.parse_args(args)

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

    logger.info(f"Processing input: {input_path}")
    run(input_path, config, output_file_path=parsed_args.output)
    logger.debug("bmod.main done")

    return 0


if __name__ == '__main__':
    sys.exit(main())
