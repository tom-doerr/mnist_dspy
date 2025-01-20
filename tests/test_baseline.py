import pytest
from mnist_dspy import MNISTClassifier
from mnist_evaluation import MNISTEvaluator

@pytest.mark.usefixtures("sample_test_data")
def test_baseline_accuracy(sample_test_data):
    """Regression test for baseline model accuracy"""
    model = MNISTClassifier(model_name="deepseek/deepseek-chat")
    evaluator = MNISTEvaluator(model_name="deepseek/deepseek-chat")
    
    accuracy = evaluator.evaluate_accuracy(sample_test_data)['accuracy']
    assert accuracy >= 0.85, f"Baseline accuracy dropped below 85% ({accuracy:.1%})"

def test_reasoner_model_initialization():
    """Test reasoner model initializes without temperature param"""
    model = MNISTClassifier(model_name="deepseek/deepseek-reasoner")
    # Verify no temperature in LM config
    assert model.predict.lm is None or "temperature" not in model.predict.lm.kwargs
