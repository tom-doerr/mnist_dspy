import dspy
from typing import List
from mnist_data import MNISTData

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
