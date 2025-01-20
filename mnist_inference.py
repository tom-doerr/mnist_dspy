import dspy
from mnist_dspy import MNISTClassifier
from typing import List, Tuple

class MNISTInference:
    def __init__(self):
        self.classifier = MNISTClassifier()
        self._configure_model()

    def _configure_model(self):
        model = dspy.OpenAI(
            model='deepseek/deepseek-chat',
            temperature=1.0
        )
        dspy.settings.configure(lm=model)

    def predict(self, pixel_matrix: str) -> str:
        return self.classifier(pixel_matrix)

    def batch_predict(self, data: List[Tuple[str, str]]) -> List[str]:
        return [self.predict(pixels) for pixels, _ in data]
