import dspy
from typing import List, Tuple
from mnist_data import MNISTData

class MNISTSignature(dspy.Signature):
    """Classify MNIST handwritten digits from their pixel matrix."""
    pixel_matrix = dspy.InputField(desc="28x28 matrix of pixel values (0-255) as text")
    digit = dspy.OutputField(desc="predicted digit from 0 to 9")

class MNISTClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(MNISTSignature)
        
    def forward(self, pixel_matrix: str) -> str:
        result = self.predict(pixel_matrix=pixel_matrix)
        return result.digit

def create_training_data() -> List[Tuple[str, str]]:
    mnist = MNISTData()
    train_data = mnist.get_training_data()
    return [(pixels, str(label)) for pixels, label in train_data[:1000]]  # Use subset for training

def create_test_data() -> List[Tuple[str, str]]:
    mnist = MNISTData()
    test_data = mnist.get_test_data()
    return [(pixels, str(label)) for pixels, label in test_data[:200]]  # Use subset for testing
