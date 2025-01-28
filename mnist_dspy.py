#!/usr/bin/env python3
import dspy
from typing import List, Tuple
from mnist_data import MNISTData

class MNISTSignature(dspy.Signature):
    """Classify MNIST handwritten numbers from their pixel matrix."""
    pixel_matrix = dspy.InputField(desc="28x28 matrix of pixel values (0-255) as text")
    digit = dspy.OutputField(desc="predicted number from 0 to 9")



class MNISTClassifier(dspy.Module):
    def __init__(self, model_name: str = "deepseek/deepseek-chat"):
        super().__init__()
        self.predict = dspy.Predict(MNISTSignature)
        lm = dspy.LM(model=model_name, temperature=1.0, cache=True)
        dspy.settings.configure(lm=lm)
        
    def forward(self, pixel_matrix: str) -> dspy.Prediction:
        return self.predict(pixel_matrix=pixel_matrix)

def create_training_data(samples: int = 1000) -> List[Tuple[str, str]]:
    print("Creating training data...")
    mnist = MNISTData()
    train_data = mnist.get_training_data()  # Only get training data
    print(f"Using {samples} samples from {len(train_data)} available training samples")

    # print("train_data:", train_data)
    # return [(pixels, str(label)) for pixels, label in train_data[:samples]]
    return train_data[:samples]

def create_test_data(samples: int = 200) -> List[dspy.Example]:
    print("Creating test data...")
    mnist = MNISTData()
    raw_test = mnist.get_test_data()
    print(f"Using {samples} samples from {len(raw_test)} available test samples")
    # test_data = [
        # dspy.Example(pixel_matrix=pixels, number=str(label)).with_inputs('pixel_matrix')  # Creating formatted test examples
        # for pixels, label in raw_test[:samples]
    # ]
    test_data = [ dspy.Example(pixel_matrix=e['pixel_matrix'], digit=e['digit']).with_inputs('pixel_matrix') for e in raw_test[:samples] ]

    
    # Print sample test data
    print("\n=== Test Data Sample ===")
    sample_ex = test_data[0]
    print(f"Sample pixel matrix shape: {len(sample_ex.pixel_matrix.split())}x{len(sample_ex.pixel_matrix.splitlines())}")
    print(f"Sample label: {sample_ex.number} (type: {type(sample_ex.number)})")
    return test_data
