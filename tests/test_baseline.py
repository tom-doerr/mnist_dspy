import pytest
from mnist_dspy import MNISTClassifier
from mnist_ensemble_booster import MNISTEnsembleBooster
from mnist_evaluation import MNISTEvaluator
from mnist_ensemble_booster import create_training_data
from mnist_data import MNISTData

@pytest.fixture
def sample_ensemble():
    """Fixture providing a trained ensemble booster with 3 iterations"""
    booster = MNISTBooster(iterations=3)
    booster.raw_data = create_training_data(samples=100)
    booster.test_pool = create_training_data(samples=50)
    booster.train_iteration(0)
    booster.train_iteration(1)
    booster.train_iteration(2)
    return booster

@pytest.mark.usefixtures("sample_test_data")
@pytest.mark.baseline
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
    
    # Test individual classifier output type
    single_result = sample_ensemble.classifiers[0](pixel_matrix=test_sample.pixel_matrix)
    assert hasattr(single_result, 'digit'), "Classifier should return Prediction object"
    assert isinstance(single_result.digit, str), "Digit attribute should be string"
    
    # Test ensemble voting
    _, voting_data = sample_ensemble.evaluate_ensemble()
    ensemble_pred = voting_data[hash(test_sample.pixel_matrix)]['majority']
    
    assert isinstance(ensemble_pred, str), "Ensemble prediction should be string digit"
    assert ensemble_pred in {str(i) for i in range(10)}, "Invalid digit prediction"

def test_hard_example_tracking(sample_ensemble):
    """Verify hard examples are being tracked across iterations"""
    assert len(sample_ensemble.hard_examples) > 0, "Should track challenging examples"
    assert 2 in sample_ensemble.misclassification_history, "Should track iteration history"

def test_classifier_output_format(sample_ensemble):
    """Verify all classifiers return Prediction objects with digit attribute"""
    test_input = "0 0 0 0 " + "0 "*(28*28-4)  # Minimal valid input
    
    for i, clf in enumerate(sample_ensemble.classifiers):
        result = clf(pixel_matrix=test_input)
        assert hasattr(result, 'digit'), f"Classifier {i} missing 'digit' attribute"
        assert isinstance(result.digit, str), f"Classifier {i} digit should be string"
        assert result.digit in {str(n) for n in range(10)}, f"Classifier {i} invalid digit: {result.digit}"

def test_confidence_score_validation(sample_ensemble):
    """Verify confidence scores are within valid range and make sense"""
    test_input = "0 0 0 0 " + "0 "*(28*28-4)  # Minimal valid input
    
    for clf in sample_ensemble.classifiers:
        result = clf(pixel_matrix=test_input)
        assert hasattr(result, 'confidence'), "Prediction missing confidence score"
        confidence = float(result.confidence)
        assert 0 <= confidence <= 1, f"Invalid confidence value {confidence}"
        
        # If confidence is very low, check if it's an uncertain prediction
        if confidence < 0.5:
            assert result.digit in {'0', '1', '5'}, "Low confidence should correlate with ambiguous digits"

def test_data_loader_output_format():
    """Verify MNIST data loader returns correctly formatted examples"""
    mnist = MNISTData()
    train_data, val_data = mnist.get_training_data()
    
    # Check training data
    for pixels, label in train_data[:100]:  # Sample first 100 examples
        assert isinstance(pixels, str), "Pixel data should be string representation"
        assert len(pixels.split()) == 784, "Incorrect number of pixel values"
        assert label in range(10), "Invalid label value"
        
    # Check validation data
    assert len(val_data) > 0, "Validation set should not be empty"
    assert len(train_data) > len(val_data), "Validation set should be smaller than training set"

