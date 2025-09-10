import logging
import pytest

import bmod.main

logger = logging.getLogger(__name__)


def test_call_cmd_no_option():
    rc = bmod.main.main([])
    assert rc == 1


def test_call_cmd_help():
    with pytest.raises(SystemExit) as e:
        bmod.main.main(['-h'])
    assert e.value.code == 0


def test_call_cmd_version():
    with pytest.raises(SystemExit) as e:
        bmod.main.main(['-V'])
    assert e.value.code == 0
