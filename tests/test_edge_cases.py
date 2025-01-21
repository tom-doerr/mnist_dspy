import pytest
from mnist_ensemble_booster import MNISTEnsembleBooster
from dspy import Example

@pytest.fixture
def edge_case_8():
    """Example that was failing with '8' label"""
    return Example({
        'pixel_matrix': '0 0 0 0 0 0 0 0 0 77 244 253 204 145 234 254 112 0 0 0 0 0 0 0 0 0\n' 
                        '0 0 0 0 0 0 0 0 0 0 0 0 80 213 254 254 254 160 7 0 0 0 0 0 0 0 0 0\n' 
                        '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n' 
                        '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n' 
                        '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0',
        'digit': '8'
    })

@pytest.fixture
def edge_case_2():
    """Example that was failing with '2' label"""
    return Example({
        'pixel_matrix': '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n'
                        '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n'
                        '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 89 128 253 255 168 0 0 0 0 0\n'
                        '0 0 0 0 0 0 0 0 38 131 246 242 167 27 18 0 0 0 0 0 0 0 0 0 0 0 0 0\n'
                        '0 0 0 0 0 0 32 222 252 208 173 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0',
        'digit': '2'
    })

def test_classifier_handles_edge_cases(sample_ensemble, edge_case_8, edge_case_2):
    """Verify all classifiers handle problematic edge cases without errors"""
    for clf in sample_ensemble.classifiers:
        # Test edge case 8
        result = clf(pixel_matrix=edge_case_8.pixel_matrix)
        assert hasattr(result, 'digit'), "Classifier missing digit attribute"
        assert result.digit in {str(i) for i in range(10)}, f"Invalid prediction: {result.digit}"
        
        # Test edge case 2
        result = clf(pixel_matrix=edge_case_2.pixel_matrix)
        assert hasattr(result, 'digit'), "Classifier missing digit attribute"
        assert result.digit in {str(i) for i in range(10)}, f"Invalid prediction: {result.digit}"

def test_ensemble_handles_edge_cases(sample_ensemble, edge_case_8, edge_case_2):
    """Verify ensemble consensus on known problematic examples"""
    # Evaluate ensemble on both edge cases
    eval_data = [edge_case_8, edge_case_2]
    acc, results = sample_ensemble.evaluate_ensemble(eval_data)
    
    # Check majority votes are valid digits
    for ex in eval_data:
        prediction = results[hash(ex.pixel_matrix)]['majority']
        assert prediction in {str(i) for i in range(10)}, f"Invalid ensemble prediction: {prediction}"
    
    # Verify edge cases are tracked as hard examples
    assert any(ex.pixel_matrix in (e.pixel_matrix for e in sample_ensemble.hard_examples) 
              for ex in eval_data), "Edge cases should be tracked as hard examples"
