#!/usr/bin/env python3
from typing import List, Tuple
from tqdm import tqdm
from mnist_inference import MNISTInference
from mnist_dspy import create_test_data

class MNISTEvaluator:
    def __init__(self):
        self.inference = MNISTInference()

    def evaluate_accuracy(self, test_data: List[Tuple[str, str]]) -> float:
        correct = 0
        total = len(test_data)
        
        with tqdm(test_data, desc="Evaluating", unit="sample") as pbar:
            for i, example in enumerate(pbar):
                # Extract data from dspy.Example
                pixels = example.pixel_matrix
                true_label = example.digit
                
                # Make prediction
                predicted = self.inference.predict(pixels)
                
                # Print detailed comparison
                print(f"\nSample {i+1}/{total}")
                print(f"True label: {true_label}")
                print(f"Predicted: {predicted}")
                print(f"Input preview:\n{pixels[:100]}...")
                
                # Update accuracy
                if predicted == true_label:
                    correct += 1
                    print("✅ Correct prediction")
                else:
                    print("❌ Incorrect prediction")
                
                # Update progress
                current_accuracy = correct / (i + 1)
                pbar.set_postfix({"accuracy": f"{current_accuracy:.2%}"})
                
        return correct / total

    def run_evaluation(self) -> float:
        test_data = create_test_data()
        return self.evaluate_accuracy(test_data)

if __name__ == "__main__":
    evaluator = MNISTEvaluator()
    accuracy = evaluator.run_evaluation()
    print(f"Model accuracy: {accuracy:.2%}")
