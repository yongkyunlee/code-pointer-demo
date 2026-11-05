"""Shared recording produced by the installed, pinned Backend.AI scheduler."""

import pytest

from backend_ai_intro.experiment import run_experiment


@pytest.fixture(scope="session")
def recording():
    return run_experiment()
