#!/usr/bin/env python3
from typing import List, Tuple
from tqdm import tqdm
from mnist_inference import MNISTInference
from mnist_dspy import create_test_data

class MNISTEvaluator:
    def __init__(self, model_name: str = "deepseek/deepseek-chat", no_cache: bool = False, num_threads: int = 100):
        self.inference = MNISTInference(model_name=model_name, no_cache=no_cache)
        self.num_threads = num_threads

    def evaluate_accuracy(self, test_data: List[Tuple[str, str]]) -> float:
        correct = 0
        total = len(test_data)
        
        with tqdm(test_data, desc="Evaluating", unit="sample") as pbar:
            dspy.configure(lm=dspy.LM(self.inference.model_name, cache=True, num_threads=self.num_threads))
            for i, example in enumerate(pbar):
                # Extract data from dspy.Example
                pixels = example.pixel_matrix
                true_label = example.digit
                
                # Make prediction
                predicted = self.inference.predict(pixels)
                
                # Handle booster model's majority voting
                if isinstance(predicted, list):
                    predicted = max(set(predicted), key=predicted.count)
                
                # Update accuracy
                if predicted == true_label:
                    correct += 1
                
                # Update progress
                current_accuracy = correct / (i + 1)
                pbar.set_postfix({"accuracy": f"{current_accuracy:.2%}"})
                pbar.update()
                
        return correct / total

    def run_evaluation(self) -> float:
        test_data = create_test_data()
        return self.evaluate_accuracy(test_data)

if __name__ == "__main__":
    evaluator = MNISTEvaluator()
    accuracy = evaluator.run_evaluation()
    print(f"Model accuracy: {accuracy:.2%}")
