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
            for i, (pixels, true_label) in enumerate(pbar):
                predicted = self.inference.predict(pixels)
                print(f"True label: {true_label}, Predicted: {predicted}")
                if predicted == true_label:
                    correct += 1
                
                # Update progress bar with current accuracy
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
