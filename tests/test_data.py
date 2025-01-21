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

def test_data_augmentation_consistency():
    """Verify that data augmentation produces valid variations"""
    mnist = MNISTData()
    original = mnist.get_training_data()[0][0]  # Get first training example
    augmented = mnist.augment_data([original])[0]  # Now uses the simplified version
    
    # Check basic properties
    assert isinstance(augmented, str), "Augmented data should be string"
    assert len(augmented.split()) == 784, "Incorrect number of pixels"
    
    # Check values are within valid range
    pixels = [int(p) for p in augmented.split()]
    assert all(0 <= p <= 255 for p in pixels), "Invalid pixel values in augmented data"

def test_preprocessing_consistency():
    """Verify pixel values are properly normalized and formatted"""
    mnist = MNISTData()
    raw_data = mnist.get_test_data()[:1]  # Single test example
    
    for pixels, _ in raw_data:
        pixel_values = [int(p) for p in pixels.split()]
        assert all(0 <= v <= 255 for v in pixel_values), "Pixel values out of range"
        assert len(pixel_values) == 784, "Incorrect number of pixels"
        assert sum(pixel_values) > 0, "Blank image should not exist in test set"
