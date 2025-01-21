import random
import dspy
from typing import List, Dict
from mnist_dspy import MNISTClassifier, create_training_data, create_test_data
from mnist_data import MNISTData
from mnist_evaluation import MNISTEvaluator
from mnist_data import MNISTData

class MNISTEnsemble:
    """Manages ensemble of classifiers and hard example tracking"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.classifiers: List[MNISTClassifier] = []
        self.hard_examples: List[dspy.Example] = []

    def _get_hard_examples(self, num_samples: int = 3) -> List[dspy.Example]:
        """Sample challenging examples using a priority hierarchy:
        1. Never-correct: Failed in ALL previous iterations
        2. Persistent: Failed in multiple but not all iterations  
        3. New errors: Recent failures
        
        Returns list of examples ordered by difficulty"""

        return random.sample(self.hard_examples, min(num_samples, len(self.hard_examples)))

    def evaluate(self) -> tuple[float, dict]:
        """Evaluate ensemble with majority voting"""
        test_data = MNISTData().get_test_data()  # Get fresh test data from source
        voting_results = {}
        
        def ensemble_predict(pixel_matrix: str) -> dspy.Prediction:
            predictions = []
            for clf in self.classifiers:
                result = clf(pixel_matrix=pixel_matrix)
                if hasattr(result, 'digit'):
                    pred = str(result.digit)
                else:
                    pred = str(result).split("'digit': '")[1].split("'")[0] if "'digit': '" in str(result) else str(result)
                predictions.append(pred)
            
            majority = max(set(predictions), key=predictions.count)
            # Store both hash and first 100 chars of pixel data for accurate matching
            key = (hash(pixel_matrix), pixel_matrix[:100])
            voting_results[key] = {
                'predictions': predictions,
                'majority': majority,
                'true_label': None,  # Initialize label field
                'correct': False
            }
            return dspy.Prediction(digit=majority)

        evaluator = MNISTEvaluator(model_name=self.model_name, num_threads=100)
        # Get accuracy from evaluator and clean up results
        accuracy = evaluator.evaluate_accuracy(test_data, predictor=ensemble_predict, 
                                             display_progress=True, display_table=0, display_summary=False)
        matched = 0
        for ex in test_data:
            # Match using the same composite key format
            key = (hash(ex.pixel_matrix), ex.pixel_matrix[:100])
            if key in voting_results:
                voting_results[key]['true_label'] = ex.digit
                voting_results[key]['correct'] = voting_results[key]['majority'] == str(ex.digit)
            else:
                # Add missing entries with full context
                voting_results[key] = {
                    'true_label': ex.digit,
                    'majority': '?',
                    'predictions': [],
                    'correct': False
                }
        return accuracy
