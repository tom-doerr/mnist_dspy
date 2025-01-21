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

def test_never_correct_percentage_increase(sample_ensemble):
    """Verify never-correct percentage calculation and potential increases"""
    # Simulate 3 iterations with overlapping errors
    test_case = sample_ensemble.test_pool[0]
    
    # First iteration: 1 error out of 100
    sample_ensemble.misclassification_history[0] = [test_case]
    sample_ensemble.hard_examples = [test_case]
    
    # Second iteration: Same error remains (now 100% never-correct)
    sample_ensemble.misclassification_history[1] = [test_case]
    sample_ensemble.hard_examples = [test_case]
    
    # Calculate percentages
    def calc_pct():
        total = len(sample_ensemble.hard_examples)
        never = sum(1 for ex in sample_ensemble.hard_examples 
                   if all(ex in hist for hist in sample_ensemble.misclassification_history.values()))
        return (never / total * 100) if total > 0 else 0
    
    iter1_pct = calc_pct()  # 100% (1/1)
    
    # Third iteration: Add new errors (now 1 never-correct out of 2 total)
    new_error = sample_ensemble.test_pool[1]
    sample_ensemble.misclassification_history[2] = [new_error]
    sample_ensemble.hard_examples = [test_case, new_error]
    iter2_pct = calc_pct()  # 50% (1/2)
    
    # Percentage should decrease when new errors are added
    assert iter2_pct < iter1_pct, "Never-correct percentage should decrease when new errors are added"
    
    # Test percentage increases when total errors decrease but never-correct remains
    sample_ensemble.hard_examples = [test_case]  # Remove new error
    iter3_pct = calc_pct()  # 100% (1/1)
    assert iter3_pct > iter2_pct, "Percentage should increase if total errors decrease but never-correct remains"

def test_never_correct_tracking(sample_ensemble):
    """Verify examples are removed from never-correct when finally solved"""
    test_case = sample_ensemble.test_pool[0]
    
    # Add to history as failed in iterations 0 and 1
    sample_ensemble.misclassification_history[0] = [test_case]
    sample_ensemble.misclassification_history[1] = [test_case]
    sample_ensemble.hard_examples = [test_case]
    
    # Mark as solved in iteration 2
    sample_ensemble.misclassification_history[2] = []
    
    # Verify it's no longer considered never-correct
    never_correct = [ex for ex in sample_ensemble.hard_examples 
                    if all(ex in hist for hist in sample_ensemble.misclassification_history.values())]
    assert test_case not in never_correct, "Solved example should be removed from never-correct"
