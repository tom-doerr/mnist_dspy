import dspy
from typing import List
from mnist_data import MNISTData
from mnist_dspy import MNISTClassifier

class MNISTBoosterV2:
    """Advanced boosting implementation with hard example tracking"""
    
    def __init__(self):
        self.hard_examples: List[dspy.Example] = []
        
    def get_hard_examples(self, test_data: List[dspy.Example], predictor) -> List[dspy.Example]:
        """Collect misclassified examples from test data"""
        self.hard_examples = []
        
        for example in test_data:
            pred = predictor(example.pixel_matrix)
            if str(pred.number) != str(example.number):
                self.hard_examples.append(example)
                
        print(f"Found {len(self.hard_examples)} hard examples from {len(test_data)} total samples")
        return self.hard_examples

if __name__ == "__main__":
    # Example usage
    booster = MNISTBoosterV2()
    test_data = MNISTData().get_test_data()[:100]  # Use first 100 test samples
    classifier = MNISTClassifier()
    
    hard_examples = booster.get_hard_examples(test_data, classifier)
    
    print("\nSample hard examples:")
    for ex in hard_examples[:3]:
        pred = classifier(ex.pixel_matrix)
        print(f"True: {ex.number} | Predicted: {pred.number}")
        print(f"Pixel matrix:\n{ex.pixel_matrix[:200]}...\n")
