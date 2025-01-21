#!/usr/bin/env python3
import dspy
from typing import List, Tuple
from mnist_data import MNISTData

class MNISTSignature(dspy.Signature):
    """Classify MNIST handwritten numbers from their pixel matrix."""
    pixel_matrix = dspy.InputField(desc="28x28 matrix of pixel values (0-255) as text")
    number = dspy.OutputField(desc="predicted number from 0 to 9")

class MNISTBooster(dspy.Module):
    """DSPy module for boosted MNIST classification using ensemble voting."""
    
    def __init__(self, model_name: str = "deepseek/deepseek-chat", boosting_iterations: int = 3, verbose: bool = False):
        super().__init__()
        self.models = [
            MNISTClassifier(model_name=model_name, verbose=verbose)
            for _ in range(boosting_iterations)
        ]
        self.verbose = verbose
        self.boosting_iterations = boosting_iterations

    def forward(self, pixel_matrix: str) -> dspy.Prediction:
        """Make ensemble prediction using majority voting."""
        predictions = []
        for model in self.models:
            pred = model(pixel_matrix=pixel_matrix)
            predictions.append(pred)
            if self.verbose:
                print(f"Model {model.model_name} prediction: {pred}")
        
        majority_vote = max(set(predictions), key=predictions.count)
        print("predictions.count:", predictions.count)
        print("predictions:", predictions)
        print("majority_vote:", majority_vote)
        dspy_prediction =  dspy.Prediction(digit=majority_vote)
        print("dspy_prediction:", dspy_prediction)
        return dspy_prediction


class MNISTClassifier(dspy.Module):
    def __init__(self, model_name: str = "deepseek/deepseek-chat", verbose: bool = False):
        super().__init__()
        self.model_name = model_name
        self.verbose = verbose
        # Configure model temperature explicitly
        if model_name == "deepseek/deepseek-reasoner":
            self.predict = dspy.Predict(MNISTSignature, lm={"temperature": None})
        elif "chat" in model_name:
            self.predict = dspy.Predict(MNISTSignature, lm={"temperature": 1.0})
        else:
            self.predict = dspy.Predict(MNISTSignature)
        
    def forward(self, pixel_matrix: str) -> dspy.Prediction:
        if self.verbose:
            print(f"\nInput pixel matrix:\n{pixel_matrix[:100]}...")  # Show first 100 chars
        result = self.predict(pixel_matrix=pixel_matrix)
        print("result:", result)
        # if self.verbose:
        if True:
            print(f"Model prediction: {result.number}")
            print(f"Full prediction result: {result}")
        return result

def create_training_data(samples: int = 1000) -> List[Tuple[str, str]]:
    print("Creating training data...")
    mnist = MNISTData()
    train_data = mnist.get_training_data()  # Only get training data
    print(f"Using {samples} samples from {len(train_data)} available training samples")
    return [(pixels, str(label)) for pixels, label in train_data[:samples]]

def create_test_data(samples: int = 200) -> List[dspy.Example]:
    print("Creating test data...")
    mnist = MNISTData()
    raw_test = mnist.get_test_data()
    print(f"Using {samples} samples from {len(raw_test)} available test samples")
    return [
        dspy.Example(pixel_matrix=pixels, number=str(label)).with_inputs('pixel_matrix')  # Creating formatted test examples
        for pixels, label in raw_test[:samples]
    ]
