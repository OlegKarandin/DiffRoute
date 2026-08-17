"""Shared test configuration.

Seeds every test: several pipeline tests assert on statistics of a randomly
initialised EdgeWeightNet, and unseeded e2e runs have produced
irreproducible results before (CLAUDE.md, Phase 1c correction #8).
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


@pytest.fixture(autouse=True)
def _seed_everything():
    torch.manual_seed(0)
