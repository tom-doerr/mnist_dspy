import random
import dspy
from typing import List, Dict
from mnist_dspy import MNISTClassifier, create_training_data, create_test_data
from mnist_evaluation import MNISTEvaluator

class MNISTEnsemble:
    """Manages ensemble of classifiers and hard example tracking"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.classifiers: List[MNISTClassifier] = []
        self.hard_examples: List[dspy.Example] = []
        self.misclassification_history: Dict[int, List[dspy.Example]] = {}
        self.raw_data = create_training_data(samples=1000)
        self.test_pool = create_test_data(samples=1000)

    def _get_hard_examples(self, num_samples: int = 3) -> List[dspy.Example]:
        """Sample challenging examples that consistently fool models"""
        if not self.hard_examples:
            return random.sample(self.raw_data, min(3, len(self.raw_data)))

        never_correct = [ex for ex in self.hard_examples 
                       if all(ex in hist for hist in self.misclassification_history.values())]
        persistent = [ex for ex in self.hard_examples if ex not in never_correct]
        
        samples = []
        samples += random.sample(never_correct, min(num_samples, len(never_correct)))
        remaining = num_samples - len(samples)
        if remaining > 0:
            samples += random.sample(persistent, min(remaining, len(persistent)))
        remaining = num_samples - len(samples)
        if remaining > 0:
            samples += random.sample(self.hard_examples, min(remaining, len(self.hard_examples)))
            
        return samples[:num_samples]

    def evaluate(self) -> tuple[float, dict]:
        """Evaluate ensemble with majority voting"""
        test_data = random.sample(self.test_pool, 1000)
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
            voting_results[hash(pixel_matrix)] = {
                'predictions': predictions,
                'majority': majority
            }
            return dspy.Prediction(digit=majority)

        evaluator = MNISTEvaluator(model_name=self.model_name, num_threads=100)
        correct = evaluator.evaluate_accuracy(test_data, predictor=ensemble_predict)
        return correct / len(test_data), voting_results
