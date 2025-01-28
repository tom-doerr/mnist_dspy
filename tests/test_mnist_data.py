#!/usr/bin/env python3
import pytest
import numpy as np
import dspy
from mnist_data import MNISTData

@pytest.mark.slow
def test_mnist_data_loading():
    """Test MNIST data loading and formatting"""
    data = MNISTData()
    train_data = data.get_training_data()
    test_data = data.get_test_data()
    
    # Test data sizes
    assert len(train_data) > 0
    assert len(test_data) > 0
    
    # Test example format
    sample = train_data[0]
    assert hasattr(sample, 'pixel_matrix')
    assert hasattr(sample, 'digit')
    assert isinstance(sample.digit, str)
    assert isinstance(sample.pixel_matrix, str)

def test_matrix_to_text_conversion():
    """Test conversion of numpy matrix to text format"""
    data = MNISTData()
    sample_matrix = np.zeros((784,))
    text = data._matrix_to_text(sample_matrix)
    
    # Check format
    lines = text.split('\n')
    assert len(lines) == 28
    assert len(lines[0].split()) == 28

def test_dataset_caching():
    """Test that dataset is properly cached"""
    data1 = MNISTData()
    data2 = MNISTData()
    
    # Verify both instances share the same cached dataset
    assert data1._dataset is data2._dataset
    assert id(data1._dataset) == id(data2._dataset)

def test_train_test_split():
    """Test train/test data separation"""
    data = MNISTData()
    train_data = data.get_training_data()
    test_data = data.get_test_data()
    
    # Verify no overlap between train and test sets
    train_matrices = [ex.pixel_matrix for ex in train_data]
    test_matrices = [ex.pixel_matrix for ex in test_data]
    assert not set(train_matrices).intersection(set(test_matrices))
