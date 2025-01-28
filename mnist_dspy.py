#!/usr/bin/env python3
import dspy
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
