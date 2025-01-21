from mnist_ensemble import MNISTEnsemble
from mnist_pipeline import MNISTPipeline
from typing import List, Dict, Tuple
import random
import dspy
from mnist_dspy import MNISTClassifier
from dspy.teleprompt import LabeledFewShot
from mnist_evaluation import MNISTEvaluator

class MNISTBooster:
    def __init__(self, iterations: int = 1, model_name: str = "deepseek/deepseek-chat"):
        self.ensemble = MNISTEnsemble(model_name)
        self.pipeline = MNISTPipeline(iterations, model_name)
        self.hard_examples = []  # Track challenging examples
        self.classifiers = []    # Store trained models
        self.model_name = model_name
        self.raw_data = []       # Training data pool
        self.test_pool = []      # Evaluation data

    def _get_hard_examples(self, num_samples: int = 3) -> List[dspy.Example]:
        """Randomly select challenging examples from our training pool"""
        if self.hard_examples:
            return random.sample(self.hard_examples, min(num_samples, len(self.hard_examples)))
        return random.sample(self.raw_data, min(num_samples, len(self.raw_data)))

    def train_iteration(self, iteration: int) -> float:
        """Train a single boosting iteration using a 3-phase approach:
        1. Sample hardest examples from previous iterations
        2. Train new classifier on these challenging cases
        3. Update tracking of persistently misclassified digits
        
        Returns accuracy on validation subset"""
        # Phase 1: Select challenging examples
        examples = self._get_hard_examples(3)  # Get top 3 hardest
        
        # Phase 2: Train specialized classifier
        classifier = MNISTClassifier(model_name=self.model_name)
        optimized = LabeledFewShot(k=len(examples)).compile(
            classifier, 
            trainset=examples
        )
        self.classifiers.append(optimized)
        
        # Phase 3: Identify persistent errors
        evaluator = MNISTEvaluator(model_name=self.model_name)
        eval_data = self.test_pool[:100]  # Fixed evaluation set
        accuracy = evaluator.evaluate_accuracy(eval_data) / len(eval_data)
        
        # Update hard examples with current model's errors
        current_errors = [
            ex for ex in eval_data 
            if ex.digit != evaluator.inference.predict(ex.pixel_matrix)
        ]
        # Merge with existing hard examples, keeping most frequent
        self.hard_examples = (self.hard_examples + current_errors)[:20]
        
        return accuracy

    def evaluate_ensemble(self) -> Tuple[float, Dict]:
        """Final evaluation with majority voting using threaded evaluation.
        
        The voting system works by:
        1. Each classifier makes independent predictions
        2. Majority vote determines final ensemble prediction
        3. Track prediction history for error analysis
        4. Use parallel evaluation for speed
        
        Returns tuple of (accuracy, detailed_voting_results)"""
        test_data = random.sample(self.test_pool, 1000)
        voting_results = {}  # Stores {input_hash: {predictions: [], majority: str}}
        
        # Create evaluator with high parallelism
        evaluator = MNISTEvaluator(model_name=self.model_name, num_threads=100)
        
        # Define threaded evaluation function
        def ensemble_predict(pixel_matrix: str) -> dspy.Prediction:
            """Make ensemble prediction for a single image matrix"""
            # Collect predictions from all classifiers
            # Safely get predictions from classifiers and extract digit values
            predictions = []
            for clf in self.classifiers:
                result = clf(pixel_matrix=pixel_matrix)
                if hasattr(result, 'digit'):
                    pred = str(result.digit)
                else:  # Handle raw string output if Prediction wrapper fails
                    pred = str(result).split("'digit': '")[1].split("'")[0] if "'digit': '" in str(result) else str(result)
                predictions.append(pred)
            
            # Get majority vote
            majority = max(set(predictions), key=predictions.count)
            
            # Store results with input hash for analysis
            voting_results[hash(pixel_matrix)] = {
                'predictions': predictions,
                'majority': majority
            }
            return dspy.Prediction(digit=majority)
            
        # Run parallel evaluation
        correct = evaluator.evaluate_accuracy(test_data, predictor=ensemble_predict)
        
        # Print predictions and labels
        print("\nPredictions vs Labels:")
        for ex in test_data:
            votes = voting_results[hash(ex.pixel_matrix)]
            print(f"True: {ex.digit} -> Predictions: {votes['predictions']} -> Majority: {votes['majority']}")
                
        return correct / len(test_data), voting_results

    def run(self):
        """Execute full boosting pipeline"""
        self.pipeline.run_training(self.ensemble)
        final_acc, results = self.ensemble.evaluate()
        self.pipeline.report_results(final_acc, results)

if __name__ == "__main__":
    booster = MNISTBooster(iterations=10)
    booster.run()
