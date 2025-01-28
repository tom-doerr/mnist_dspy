#!/usr/bin/env python3
import pytest
import dspy
from mnist_dspy import MNISTClassifier, MNISTBooster

def test_mnist_classifier_prediction():
    """Test basic prediction functionality"""
    classifier = MNISTClassifier(model_name="deepseek/deepseek-chat")
    sample_input = "0 0 0\n0 1 0\n0 0 0"  # Minimal test matrix
    
    prediction = classifier(pixel_matrix=sample_input)
    assert hasattr(prediction, 'digit')
    assert prediction.digit in [str(i) for i in range(10)]

def test_ensemble_voting():
    """Test ensemble voting logic"""
    booster = MNISTBooster(boosting_iterations=3)
    sample_input = "0 0 0\n0 1 0\n0 0 0"
    
    prediction = booster(pixel_matrix=sample_input)
    assert hasattr(prediction, 'digit')
    assert prediction.digit in [str(i) for i in range(10)]

def test_invalid_model_name():
    """Test handling of invalid model name"""
    with pytest.raises(Exception):
        MNISTClassifier(model_name="invalid_model")

def test_empty_input():
    """Test handling of empty input"""
    classifier = MNISTClassifier()
    with pytest.raises(ValueError):
        classifier(pixel_matrix="")

def test_malformed_matrix():
    """Test handling of malformed matrix input"""
    classifier = MNISTClassifier()
    with pytest.raises(ValueError):
        classifier(pixel_matrix="not a matrix")
