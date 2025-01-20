import pytest
from mnist_data import MNISTData

@pytest.fixture
def sample_training_data():
    mnist = MNISTData()
    train, val = mnist.get_training_data(validation_ratio=0.1)
    return train[:100], val[:20]  # Small subsets for fast testing

@pytest.fixture
def sample_test_data():
    mnist = MNISTData()
    return mnist.get_test_data()[:50]  # Small subset for fast testing
