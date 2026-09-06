"""Shared test configuration.

Seeds every test: several pipeline tests assert on statistics of a randomly
initialised AllocationHead, and unseeded e2e runs have produced
irreproducible results before (docs/investigations/CHANGELOG.md#correction-1c-8).
"""
import sys
from pathlib import Path

import pytest
import torch

# data/ is not a package (no __init__.py, not installed editable), so
# test_generate_qot_dataset.py's `from generate_qot_dataset import ...`
# needs it on sys.path explicitly. Genuinely needed, unlike the diffopt/
# sys.path inserts removed elsewhere in tests/ (diffopt is installed
# editable and importable without this).
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

# Same story for scripts/: it is a namespace package (no __init__.py), so
# `from scripts._common import ...` already resolves off the rootdir, but
# `from calibrate_lambda_dev import ...` — importing a SCRIPT as a module, the
# way tests/test_calibrate_lambda_dev.py unit-tests its arithmetic — needs
# scripts/ itself on sys.path.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


@pytest.fixture(autouse=True)
def _seed_everything():
    torch.manual_seed(0)
