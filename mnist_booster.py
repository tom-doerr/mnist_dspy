from mnist_ensemble import MNISTEnsemble
from mnist_pipeline import MNISTPipeline
from typing import List, Dict, Tuple
import random
import dspy
from mnist_dspy import MNISTClassifier
from dspy.teleprompt import LabeledFewShot
from mnist_evaluation import MNISTEvaluator

class MNISTBooster:
    def __init__(self, iterations: int = 10, model_name: str = "deepseek/deepseek-chat"):
        self.ensemble = MNISTEnsemble(model_name)
        self.pipeline = MNISTPipeline(iterations, model_name)

    def _get_hard_examples(self, num_samples: int = 3) -> List[dspy.Example]:
        """Sample challenging examples that consistently fool models:
        - Frequently misclassified digits (e.g. 4 vs 9, 7 vs 1)
        - Ambiguous handwritten shapes
        - Edge cases with unusual rotations/sizes
        Prioritizes examples that persist across iterations"""
        if not self.hard_examples:
            return random.sample(self.raw_data, min(3, len(self.raw_data)))
            
        # Explicitly prioritize never-correct examples first
        never_correct = [ex for ex in self.hard_examples 
                       if all(ex in hist for hist in self.misclassification_history.values())]
        persistent = [ex for ex in self.hard_examples if ex not in never_correct]
        
        # Sample priority order: never-correct -> persistent -> new
        samples = []
        samples += random.sample(never_correct, min(num_samples, len(never_correct)))
        remaining = num_samples - len(samples)
        if remaining > 0:
            samples += random.sample(persistent, min(remaining, len(persistent)))
        remaining = num_samples - len(samples)
        if remaining > 0:
            samples += random.sample(self.hard_examples, min(remaining, len(self.hard_examples)))
            
        return samples[:num_samples]  # Ensure exact sample count

    def train_iteration(self, iteration: int) -> float:
        """Train a single iteration classifier"""
        # Sample hard examples + random baseline
        fewshot_examples = self._get_hard_examples(3)
        random.shuffle(fewshot_examples)
        
        # Create and train optimized classifier
        classifier = MNISTClassifier(model_name=self.model_name)
        optimizer = LabeledFewShot(k=len(fewshot_examples))
        optimized_classifier = optimizer.compile(classifier, trainset=fewshot_examples)
        
        # Store both the compiled classifier and its predictor module
        self.classifiers.append(optimized_classifier)
        optimized_predictor = optimized_classifier.predict
        
        # Use same 100 samples repeatedly to find hard cases
        eval_data = self.test_pool[:100]  # Fixed set for consistent evaluation
        # Evaluate using the actual optimized predictor from this iteration
        evaluator = MNISTEvaluator(model_name=self.model_name, num_threads=100)
        evaluator.inference.classifier.predict = optimized_predictor  # Use current iteration's optimized predictor
        accuracy = evaluator.evaluate_accuracy(eval_data) / len(eval_data)
        
        # Find persistent hard cases by checking against original errors
        current_hard = [ex for ex in eval_data if ex.digit != evaluator.inference.predict(ex.pixel_matrix)]
        if self.hard_examples:
            # Keep examples that were hard in previous OR current iteration
            # Track example types:
            # - Never-correct: Failed in ALL iterations (most critical)
            # - Persistent: Failed in SOME previous iterations
            # - New: First-time failures (current iteration only)
            never_correct = [ex for ex in current_hard 
                           if all(ex in hist for hist in self.misclassification_history.values())]
            
            persistent_hard = [ex for ex in self.hard_examples  # Failed before AND now
                             if ex in current_hard]
            new_hard = list(set(self.hard_examples + current_hard))  # Combine history
            self.hard_examples = never_correct + persistent_hard + new_hard[:20]
        else:
            self.hard_examples = current_hard
            new_hard = current_hard
        self.misclassification_history[iteration] = new_hard
        
        return accuracy

    def evaluate_ensemble(self) -> Tuple[float, Dict]:
        """Final evaluation with majority voting using threaded evaluation"""
        test_data = random.sample(self.test_pool, 1000)
        voting_results = {}
        
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
                
        return correct / len(test_data), voting_results

    def run(self):
        """Execute full boosting pipeline"""
        self.pipeline.run_training(self.ensemble)
        final_acc, results = self.ensemble.evaluate()
        self.pipeline.report_results(final_acc, results)

if __name__ == "__main__":
    booster = MNISTBooster(iterations=10)
    booster.run()
