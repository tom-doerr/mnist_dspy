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
        print("\n=== DEBUG: Creating evaluator ===")
        print(f"Test data size: {len(test_data)}")
        print(f"First example keys: {vars(test_data[0]).keys() if test_data else 'No data'}")
        
        def metric_fn(example, pred):
            true_label = str(example.digit)
            pred_label = str(pred.digit) if hasattr(pred, 'digit') else str(pred)
            match = true_label == pred_label
            
            # Debug print for first 10 examples
            if example._index < 10:  # Using internal _index added by Evaluate
                print(f"\n- Example {example._index} -")
                print(f"True: {true_label} ({type(true_label)})")
                print(f"Pred: {pred_label} ({type(pred_label)})")
                print(f"Match: {match}")
                if not match:
                    print(f"Input preview:\n{str(example.pixel_matrix)[:200]}...")
            
            return match

        evaluator = Evaluate(
            devset=test_data,
            metric=metric_fn,
            num_threads=self.num_threads,
            display_progress=display_progress,
            display_table=5
        )
        print("Evaluator created with custom metric function")
        
        # Use custom predictor if provided, else default classifier
        predictor = predictor or self.inference.classifier
        
        # Configure LM with caching and threading
        dspy.configure(lm=dspy.LM(self.inference.model_name, cache=True, num_threads=self.num_threads))
        
        # Run evaluation using DSPy's Evaluate utility
        print("\n=== DEBUG: Starting evaluation ===")
        accuracy = evaluator(predictor) / 100  # Convert percentage to ratio
        
        print("\n=== DEBUG: Evaluation summary ===")
        print(f"Total examples: {len(test_data)}")
        print(f"Correct predictions: {int(accuracy * len(test_data))}")
        print(f"Accuracy: {accuracy:.2%}")
        
        if accuracy == 0.0:
            print("\n!!! WARNING: All predictions incorrect !!!")
            print("First incorrect prediction details:")
            first_ex = test_data[0]
            pred = predictor(first_ex.pixel_matrix)
            print(f"True: {first_ex.digit} | Pred: {pred.digit}")
            print(f"Input matrix:\n{str(first_ex.pixel_matrix)[:500]}...")
            
        return accuracy

    def run_evaluation(self) -> float:
        test_data = create_test_data()
        return self.evaluate_accuracy(test_data)

if __name__ == "__main__":
    evaluator = MNISTEvaluator()
    accuracy = evaluator.run_evaluation()
    print(f"Model accuracy: {accuracy:.2%}")
