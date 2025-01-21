#!/usr/bin/env python3
import dspy
from typing import List, Tuple
from tqdm import tqdm
from dspy.evaluate import Evaluate
from mnist_inference import MNISTInference
from mnist_dspy import create_test_data

class MNISTEvaluator:
    def __init__(self, model_name: str = "deepseek/deepseek-chat", no_cache: bool = False, num_threads: int = 100):
        self.inference = MNISTInference(model_name=model_name, no_cache=no_cache)
        self.num_threads = num_threads

    def evaluate_accuracy(self, test_data: List[Tuple[str, str]], predictor=None, 
                        display_progress: bool = True, display_table: int = 0, 
                        display_summary: bool = False) -> float:
        # Print sample predictions
        print("\nSample predictions:")
        for ex in test_data[:3]:
            pred = predictor(ex.pixel_matrix)
            print(f"Input:\n{ex.pixel_matrix[:100]}...")
            print(f"True: {ex.digit} | Predicted: {pred}\n")
        evaluator = Evaluate(
            devset=test_data,
            metric=lambda example, pred: example.digit == pred,
            num_threads=self.num_threads,
            display_progress=display_progress,
            display_table=5
        )
        
        # Use custom predictor if provided, else default classifier
        predictor = predictor or self.inference.classifier
        
        # Configure LM with caching and threading
        dspy.configure(lm=dspy.LM(self.inference.model_name, cache=True, num_threads=self.num_threads))
        
        # Run evaluation using DSPy's Evaluate utility
        accuracy = evaluator(predictor) / 100  # Convert percentage to ratio
        return accuracy

    def run_evaluation(self) -> float:
        test_data = create_test_data()
        return self.evaluate_accuracy(test_data)

if __name__ == "__main__":
    evaluator = MNISTEvaluator()
    accuracy = evaluator.run_evaluation()
    print(f"Model accuracy: {accuracy:.2%}")
