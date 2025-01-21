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
        """Sample challenging examples using a priority hierarchy:
        1. Never-correct: Failed in ALL previous iterations
        2. Persistent: Failed in multiple but not all iterations  
        3. New errors: Recent failures
        
        Returns list of examples ordered by difficulty"""
        if not self.hard_examples:
            return random.sample(self.raw_data, min(3, len(self.raw_data)))

        # Classify errors by persistence level
        never_correct = [
            ex for ex in self.hard_examples 
            if all(ex in hist for hist in self.misclassification_history.values())
        ]
        persistent = [
            ex for ex in self.hard_examples 
            if sum(ex in hist for hist in self.misclassification_history.values()) > 1
        ]
        
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
        total = len(test_data)
        correct = evaluator.evaluate_accuracy(test_data, predictor=ensemble_predict, 
                                            display_progress=True, display_table=0, display_summary=False)
        # Calculate actual correct count from voting results
        actual_correct = sum(1 for v in voting_results.values() if v.get('correct', False))
        accuracy = actual_correct / total if total > 0 else 0.0
        
        print(f"\nDebug - Evaluator reported: {correct}/{total}")
        print(f"Debug - Actual verified correct: {actual_correct}/{total}")
        
        # Print first 5 predictions with true labels
        print("\nSample predictions:")
        for idx, (key, result) in enumerate(list(voting_results.items())[:5]):
            print(f"Ex {idx+1}: Pred {result['majority']} | True {result['true_label']} | Votes {result['predictions']}")
        
        # Store true labels in voting results for analysis
        matched = 0
        for ex in test_data:
            # Create more unique key with hash + first 10 pixels
            pixel_prefix = ex.pixel_matrix[:50]
            key = hash(pixel_prefix)
            if key in voting_results:
                voting_results[key]['true_label'] = ex.digit
                # Also store actual prediction vs truth
                voting_results[key]['correct'] = voting_results[key]['majority'] == str(ex.digit)
                
        return accuracy, voting_results
