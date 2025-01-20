#!/usr/bin/env python3
import dspy
from mnist_dspy import MNISTClassifier
from typing import List, Tuple

class MNISTInference:
    def __init__(self, model_name: str = "deepseek/deepseek-chat", no_cache: bool = False):
        self.classifier = MNISTClassifier(model_name)
        self._configure_model(model_name, no_cache)

    def _configure_model(self, model_name: str, no_cache: bool = False):
        model = dspy.LM(
            model=model_name,
            temperature=1.0,
            cache=not no_cache
        )
        dspy.settings.configure(lm=model)

    def predict(self, pixel_matrix: str) -> str:
        return self.classifier(pixel_matrix)

    def batch_predict(self, data: List[Tuple[str, str]]) -> List[str]:
        return [self.predict(pixels) for pixels, _ in data]
