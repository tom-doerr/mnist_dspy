#!/usr/bin/env python3
import pytest
import numpy as np
import dspy
from mnist_data import MNISTData

@pytest.fixture
def sample_matrix():
    """Return a sample 28x28 matrix"""
    return np.zeros((784,)).reshape(28, 28)

@pytest.fixture
def mnist_data():
    """Return a MNISTData instance"""
    return MNISTData()

@pytest.fixture
def sample_examples():
    """Return a list of sample dspy.Example objects"""
    return [
        dspy.Example(pixel_matrix="0 0\n0 0", digit="0").with_inputs('pixel_matrix'),
        dspy.Example(pixel_matrix="1 1\n1 1", digit="1").with_inputs('pixel_matrix')
    ]
