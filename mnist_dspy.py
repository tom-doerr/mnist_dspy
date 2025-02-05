#!/usr/bin/env python3
import dspy
from mnist_data import MNISTData
import time

class MNISTSignature(dspy.Signature):
    """Classify MNIST handwritten numbers from their pixel matrix."""
    pixel_matrix = dspy.InputField(desc="784-dimensional numpy array of pixel values between 0 and 1")
    digit = dspy.OutputField(desc="predicted number from 0 to 9")

class MNISTClassifier(dspy.Module):
    def __init__(self, model_name: str = "deepseek/deepseek-chat", cache: bool = True):
        super().__init__()
        self.predict = dspy.Predict(MNISTSignature)
        lm = dspy.LM(model=model_name, temperature=1.0, cache=cache, num_retries=12)
        dspy.settings.configure(lm=lm)
        
    def forward(self, pixel_matrix: str) -> dspy.Prediction:
        try:
            # return self.predict(pixel_matrix=pixel_matrix)
            request_start = time.time()
            prediction = self.predict(pixel_matrix=pixel_matrix)
            print(f"Prediction took {time.time() - request_start:.3f} seconds")
            # print("prediction:", prediction)
            return prediction
        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            # return dspy.Prediction(digit='0')  # Return default prediction on error
            return dspy.Prediction(digit=None) 
            # return None
