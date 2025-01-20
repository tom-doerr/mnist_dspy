import pytest
import dspy
from mnist_data import MNISTData

@pytest.fixture
def sample_training_data():
    mnist = MNISTData()
    train, val = mnist.get_training_data(validation_ratio=0.1)
    return train[:100], val[:20]  # Small subsets for fast testing

@pytest.fixture
def sample_test_data():
    mnist = MNISTData()
    raw_data = mnist.get_test_data()[:50]  # Small subset for fast testing
    return [dspy.Example(pixel_matrix=pixels, digit=str(label)).with_inputs('pixel_matrix') 
            for pixels, label in raw_data]
