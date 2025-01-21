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
        # Print sample predictions with more debug info
        print("\n=== DEBUG: Sample predictions ===")
        for i, ex in enumerate(test_data[:3]):
            print(f"\n- Example {i+1} -")
            print(f"Input type: {type(ex.pixel_matrix)}")
            print(f"Input length: {len(ex.pixel_matrix) if hasattr(ex, 'pixel_matrix') else 'N/A'}")
            print(f"True label type: {type(ex.number)}")
            
            try:
                pred = predictor(ex.pixel_matrix)
                print(f"Raw prediction: {pred}")
                print(f"Predicted label type: {type(pred.number) if hasattr(pred, 'number') else type(pred)}")
                print(f"True: {ex.number} | Predicted: {pred.number}")
                print(f"Input preview:\n{str(ex.pixel_matrix)[:100]}...")
            except Exception as e:
                print(f"Prediction failed: {str(e)}")
        print("\n=== DEBUG: Creating evaluator ===")
        print(f"Test data size: {len(test_data)}")
        print(f"First example keys: {vars(test_data[0]).keys() if test_data else 'No data'}")
        
        def metric_fn(example, pred):
            true_label = str(example.number)
            pred_label = str(pred.number) if hasattr(pred, 'number') else str(pred)
            match = true_label == pred_label
            
            # Debug print for first 10 examples
            if example.idx < 10:  # Track index via example's idx field
                print(f"\n- Example {example._index} -")
                print(f"True: {true_label} ({type(true_label)})")
                print(f"Pred: {pred_label} ({type(pred_label)})")
                print(f"Match: {match}")
                if not match:
                    print(f"Input preview:\n{str(example.pixel_matrix)[:200]}...")
            
            return match

        # Create indexed examples for tracking
        # Create new examples with idx field added
        indexed_data = []
        for i, ex in enumerate(test_data):
            new_ex = dspy.Example(pixel_matrix=ex.pixel_matrix, number=ex.number, idx=i).with_inputs('pixel_matrix')
            indexed_data.append(new_ex)
        
        evaluator = Evaluate(
            devset=indexed_data,
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
            print(f"True: {first_ex.number} | Pred: {pred.number}") 
            print(f"Input matrix:\n{str(first_ex.pixel_matrix)[:500]}...")
            
        return accuracy

    def run_evaluation(self) -> float:
        test_data = create_test_data()
        return self.evaluate_accuracy(test_data)

if __name__ == "__main__":
    evaluator = MNISTEvaluator()
    accuracy = evaluator.run_evaluation()
    print(f"Model accuracy: {accuracy:.2%}")
