import pytest
from mnist_dspy import MNISTClassifier
from mnist_evaluation import MNISTEvaluator

@pytest.mark.usefixtures("sample_test_data")
def test_baseline_accuracy(sample_test_data):
    """Regression test for baseline model accuracy"""
    model = MNISTClassifier(model_name="deepseek/deepseek-chat")
    evaluator = MNISTEvaluator(model_name="deepseek/deepseek-chat")
    
    accuracy = evaluator.evaluate_accuracy(sample_test_data)
    assert accuracy >= 0.15, f"Baseline accuracy dropped below 15% ({accuracy:.0%})"

