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

    def evaluate(self) -> float:
        """Evaluate ensemble with majority voting"""
        test_data = MNISTData().get_test_data()  # Get fresh test data from source
        def ensemble_predict(pixel_matrix: str) -> dspy.Prediction:
            predictions = []
            for clf in self.classifiers:
                result = clf(pixel_matrix=pixel_matrix)
                predictions.append(str(result.digit))
            majority = max(set(predictions), key=predictions.count)
            return dspy.Prediction(digit=majority)

        evaluator = MNISTEvaluator(model_name=self.model_name, num_threads=100)
        return evaluator.evaluate_accuracy(test_data, predictor=ensemble_predict, 
                                         display_progress=True, display_table=0, display_summary=False)
