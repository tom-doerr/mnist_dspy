#!/usr/bin/env python3
from dspy.evaluate import Evaluate
from mnist_data import MNISTData
from mnist_dspy import MNISTClassifier
from typing import List
import dspy

class MNISTDSPyEvaluator:
    """Evaluates DSPy MNIST classifier with configurable thread pooling and example limits."""
    
    def __init__(self, model_name: str = "deepseek/deepseek-reasoner", num_threads: int = 100):
        self.model_name = model_name
        self.num_threads = num_threads
        self.classifier = MNISTClassifier(model_name)
        self.data = MNISTData()
        # Configure DSPy with the LM from the classifier
        dspy.configure(lm=self.classifier.predict.lm)
        # Configure DSPy with the LM from the classifier
        dspy.configure(lm=self.classifier.predict.lm)
        
    def evaluate(self, test_data: List[dspy.Example] = None, limit: int = 100) -> float:
        """Run evaluation with threaded execution and example limiting.
        
        Args:
            test_data: Optional pre-loaded test dataset
            limit: Maximum number of examples to evaluate (default: 100)
            
        Returns:
            Accuracy percentage between 0.0 and 1.0
        """
        if test_data is None:
            test_data = self.data.get_test_data()
            
        # Apply example limit while preserving original data ordering
        limited_data = test_data[:limit]
        
        evaluator = Evaluate(
            devset=limited_data,
            metric=self._accuracy_metric,
            num_threads=self.num_threads,
            display_progress=True,
            display_table=0
        )
        
        return evaluator(self.classifier)
        
    def _accuracy_metric(self, example, pred, trace=None) -> float:
        """Simple exact match accuracy metric using digit comparison."""
        return int(example.digit) == int(pred.digit)

if __name__ == "__main__":
    evaluator = MNISTDSPyEvaluator()
    accuracy = evaluator.evaluate()
    print(f"MNIST Classification Accuracy: {accuracy:.2%}")
