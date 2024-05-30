import pytest
import numpy as np
from WildfireLogan.fergusonfire import create_pairs


@pytest.fixture
def sample_data():
    return np.arange(200)


@pytest.fixture
def cycle_start_indices():
    # Create sample cycle start indices
    return [0, 100]


def test_create_pairs_no_cycles(sample_data):
    no_cycle_indices = []

    pairs = create_pairs(sample_data, no_cycle_indices)
    assert pairs.size == 0  # No pairs should be created


if __name__ == "__main__":
    pytest.main()
