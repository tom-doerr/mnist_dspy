import pytest
from mnist_dspy import MNISTClassifier, MNISTEnsembleBooster
from mnist_evaluation import MNISTEvaluator
from mnist_ensemble_booster import create_training_data

@pytest.fixture
def sample_ensemble():
    """Fixture providing a trained ensemble booster with 3 iterations"""
    booster = MNISTEnsembleBooster(iterations=3)
    booster.raw_data = create_training_data(samples=100)
    booster.test_pool = create_training_data(samples=50)
    booster.train_iteration(0)
    booster.train_iteration(1)
    booster.train_iteration(2)
    return booster

@pytest.mark.usefixtures("sample_test_data")
def test_baseline_accuracy(sample_test_data):
    """Regression test for baseline model accuracy"""
    model = MNISTClassifier(model_name="deepseek/deepseek-chat")
    evaluator = MNISTEvaluator(model_name="deepseek/deepseek-chat")
    
    accuracy = evaluator.evaluate_accuracy(sample_test_data)
    assert accuracy >= 0.15, f"Baseline accuracy dropped below 15% ({accuracy:.0%})"

def test_ensemble_improvement(sample_ensemble, sample_test_data):
    """Verify ensemble performs better than single model baseline"""
    single_model_acc = sample_ensemble.train_iteration(0)  # First model accuracy
    ensemble_acc, _ = sample_ensemble.evaluate_ensemble()
    assert ensemble_acc > single_model_acc, "Ensemble should outperform single models"

def test_ensemble_voting(sample_ensemble, sample_test_data):
    """Verify majority voting mechanism works correctly"""
    test_sample = sample_test_data[0]
    results = sample_ensemble.classifiers[0](pixel_matrix=test_sample.pixel_matrix)
    single_pred = results.digit
    
    _, voting_data = sample_ensemble.evaluate_ensemble()
    ensemble_pred = voting_data[hash(test_sample.pixel_matrix)]['majority']
    
    assert isinstance(ensemble_pred, str), "Ensemble prediction should be string digit"
    assert ensemble_pred in {str(i) for i in range(10)}, "Invalid digit prediction"

def test_hard_example_tracking(sample_ensemble):
    """Verify hard examples are being tracked across iterations"""
    assert len(sample_ensemble.hard_examples) > 0, "Should track challenging examples"
    assert 2 in sample_ensemble.misclassification_history, "Should track iteration history"

