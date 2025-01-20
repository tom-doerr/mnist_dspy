#!/usr/bin/env python3
import dspy
from typing import List, Tuple
from mnist_data import MNISTData

class MNISTSignature(dspy.Signature):
    """Classify MNIST handwritten digits from their pixel matrix."""
    pixel_matrix = dspy.InputField(desc="28x28 matrix of pixel values (0-255) as text")
    digit = dspy.OutputField(desc="predicted digit from 0 to 9")

class MNISTClassifier(dspy.Module):
    def __init__(self, model_name: str = "deepseek/deepseek-chat", verbose: bool = True):
        super().__init__()
        self.model_name = model_name
        # Configure model with temperature only for chat models
        # Only pass temperature for chat models
        if "chat" in model_name:
            self.predict = dspy.Predict(MNISTSignature, lm={"temperature": 1.0})
        else:
            self.predict = dspy.Predict(MNISTSignature)
        
    def forward(self, pixel_matrix: str) -> str:
        if self.verbose:
            print(f"\nInput pixel matrix:\n{pixel_matrix[:100]}...")  # Show first 100 chars
        result = self.predict(pixel_matrix=pixel_matrix)
        if self.verbose:
            print(f"Model prediction: {result.digit}")
            print(f"Full prediction result: {result}")
        return result.digit

def create_training_data() -> List[Tuple[str, str]]:
    print("Creating training data...")
    mnist = MNISTData()
    train_data = mnist.get_training_data()
    print(f"Using first 1000 samples from {len(train_data)} available training samples")
    return [(pixels, str(label)) for pixels, label in train_data[:1000]]  # Use subset for training

def create_test_data() -> List[Tuple[str, str]]:
    print("Creating test data...")
    mnist = MNISTData()
    test_data = mnist.get_test_data()
    print(f"Using first 200 samples from {len(test_data)} available test samples")
    return [(pixels, str(label)) for pixels, label in test_data[:200]]  # Use subset for testing
