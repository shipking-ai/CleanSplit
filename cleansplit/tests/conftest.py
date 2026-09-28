import numpy as np
import pytest

from cleansplit.audio.synthetic import make_song


@pytest.fixture(scope="session")
def song():
    """8 s synthetic six-stem song (mixture, stems)."""
    return make_song(duration_s=8.0, seed=0)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
