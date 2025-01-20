from typing import List, Tuple
from mnist_inference import MNISTInference
from mnist_dspy import create_test_data

class MNISTEvaluator:
    def __init__(self):
        self.inference = MNISTInference()

    def evaluate_accuracy(self, test_data: List[Tuple[str, str]]) -> float:
        correct = 0
        total = len(test_data)
        
        for pixels, true_label in test_data:
            predicted = self.inference.predict(pixels)
            if predicted == true_label:
                correct += 1
                
        return correct / total

    def run_evaluation(self) -> float:
        test_data = create_test_data()
        return self.evaluate_accuracy(test_data)

if __name__ == "__main__":
    evaluator = MNISTEvaluator()
    accuracy = evaluator.run_evaluation()
    print(f"Model accuracy: {accuracy:.2%}")
