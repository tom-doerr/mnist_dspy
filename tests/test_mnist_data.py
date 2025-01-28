#!/usr/bin/env python3
import pytest
from mnist_data import MNISTData

@pytest.mark.slow
def test_mnist_data_loading():
    """Test basic data loading functionality"""
    data = MNISTData()
    train_data = data.get_training_data()[:10]
    test_data = data.get_test_data()[:10]
    
    assert len(train_data) == 10
    assert len(test_data) == 10
    
    example = train_data[0]
    assert hasattr(example, 'pixel_matrix')
    assert hasattr(example, 'digit')
    assert isinstance(example.digit, str)
