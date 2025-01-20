#!/usr/bin/env python3
import random
import dspy
from typing import List, Tuple, Dict
from tqdm import tqdm
from mnist_dspy import MNISTClassifier, create_training_data, create_test_data
from mnist_evaluation import MNISTEvaluator
from dspy.teleprompt import LabeledFewShot

class MNISTEnsembleBooster:
    def __init__(self, iterations: int = 10, model_name: str = "deepseek/deepseek-chat"):
        self.iterations = iterations
        self.model_name = model_name
        self.classifiers: List[MNISTClassifier] = []
        self.hard_examples: List[dspy.Example] = []
        self.misclassification_history: Dict[int, List[dspy.Example]] = {}
        
        # Configure LM with caching
        dspy.configure(lm=dspy.LM(model_name, cache=True))
        
        # Initial training data
        self.raw_data = create_training_data(samples=1000)
        self.test_pool = create_test_data(samples=1000)

    def _get_hard_examples(self, num_samples: int = 3) -> List[dspy.Example]:
        """Sample challenging examples from misclassified pool"""
        if not self.hard_examples:
            return random.sample(self.raw_data, min(3, len(self.raw_data)))
        return random.sample(self.hard_examples, min(num_samples, len(self.hard_examples)))

    def train_iteration(self, iteration: int) -> float:
        """Train a single iteration classifier"""
        # Sample hard examples + random baseline
        fewshot_examples = self._get_hard_examples(3)
        random.shuffle(fewshot_examples)
        
        # Create classifier and optimizer
        classifier = MNISTClassifier(model_name=self.model_name)
        optimizer = LabeledFewShot(k=len(fewshot_examples))
        
        # Train with current hard examples
        optimized = optimizer.compile(classifier, trainset=fewshot_examples)
        self.classifiers.append(optimized)
        
        # Use same 100 samples repeatedly to find hard cases
        eval_data = self.test_pool[:100]  # Fixed set for consistent evaluation
        # Evaluate with 100 threads for parallel processing
        evaluator = MNISTEvaluator(model_name=self.model_name, num_threads=100)
        accuracy = evaluator.evaluate_accuracy(eval_data)
        
        # Collect misclassified examples
        new_hard = [ex for ex in eval_data if ex.digit != evaluator.inference.predict(ex.pixel_matrix)]
        self.hard_examples.extend(new_hard)
        self.misclassification_history[iteration] = new_hard
        
        return accuracy

    def evaluate_ensemble(self) -> Tuple[float, Dict]:
        """Final evaluation with majority voting"""
        test_data = random.sample(self.test_pool, 1000)
        correct = 0
        voting_results = {}
        
        for ex in tqdm(test_data, desc="Ensemble Evaluation"):
            predictions = [clf(pixel_matrix=ex.pixel_matrix).digit for clf in self.classifiers]
            majority = max(set(predictions), key=predictions.count)
            voting_results[ex.pixel_matrix] = {
                'predictions': predictions,
                'majority': majority,
                'true_label': ex.digit
            }
            if majority == ex.digit:
                correct += 1
                
        return correct / len(test_data), voting_results

    def run(self):
        """Execute full boosting pipeline"""
        print(f"Starting ensemble boosting with {self.iterations} iterations\n")
        
        for i in range(self.iterations):
            acc = self.train_iteration(i)
            remaining = len(self.hard_examples)
            print(f"Iteration {i+1}: Accuracy {acc:.2%} | Hard Examples: {remaining} ({remaining/1000:.1%})")
        
        print("\nRunning final ensemble evaluation...")
        final_acc, results = self.evaluate_ensemble()
        print(f"\nFinal Ensemble Accuracy: {final_acc:.2%}")
        
        # Calculate error reduction
        initial_errors = len(self.misclassification_history.get(0, []))
        final_errors = sum(1 for v in results.values() if v['majority'] != v['true_label'])
        print(f"Error Reduction: {initial_errors} → {final_errors} ({(initial_errors-final_errors)/initial_errors:.1%})")

if __name__ == "__main__":
    booster = MNISTEnsembleBooster(iterations=10)
    booster.run()
