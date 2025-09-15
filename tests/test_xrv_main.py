import logging
import pytest

import bmod.xrv_main

logger = logging.getLogger(__name__)


def test_call_cmd_no_args():
    with pytest.raises(SystemExit) as e:
        bmod.xrv_main.main([])
    assert e.value.code == 2   # argparse uses 2 for "bad arguments"


def test_call_cmd_help():
    with pytest.raises(SystemExit) as e:
        bmod.xrv_main.main(['-h'])
    assert e.value.code == 0


def test_call_cmd_version():
    with pytest.raises(SystemExit) as e:
        bmod.xrv_main.main(['-V'])
    assert e.value.code == 0
