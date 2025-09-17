from pathlib import Path
import logging
import pytest
import bmod.xrv_twiss_main

logger = logging.getLogger(__name__)


# Construct the absolute path to the input file
PROJECT_ROOT = Path(__file__).parent.parent  # Adjust as needed
input_file = str(PROJECT_ROOT / "res" / "xrv4000" / "spots_tr1.csv")
output_file = "spots_tr1_fits.csv"


def test_call_cmd_no_args():
    with pytest.raises(SystemExit) as e:
        bmod.xrv_twiss_main.main([])
    assert e.value.code == 2   # argparse uses 2 for "bad arguments"


def test_call_cmd_help():
    with pytest.raises(SystemExit) as e:
        bmod.xrv_twiss_main.main(['-h'])
    assert e.value.code == 0


def test_call_cmd_version():
    with pytest.raises(SystemExit) as e:
        bmod.xrv_twiss_main.main(['-V'])
    assert e.value.code == 0


def test_call_main_logic():
    assert bmod.xrv_twiss_main.main(['-vv', input_file, output_file]) == 0
