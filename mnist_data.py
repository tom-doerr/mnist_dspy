from typing import List, Tuple
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

class MNISTData:
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = self._load_data()

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        print("Loading MNIST data from OpenML...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        print(f"Loaded {len(mnist.data)} samples")
        X = mnist.data.astype(np.uint8)
        y = mnist.target.astype(np.uint8)
        print("Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        return X_train, X_test, y_train, y_test

    def _matrix_to_text(self, matrix: np.ndarray) -> str:
        return '\n'.join(' '.join(str(pixel) for pixel in row) for row in matrix.reshape(28, 28))

    def get_training_data(self) -> List[Tuple[str, int]]:
        return [(self._matrix_to_text(x), int(y)) for x, y in zip(self.X_train, self.y_train)]

    def get_test_data(self) -> List[Tuple[str, int]]:
        return [(self._matrix_to_text(x), int(y)) for x, y in zip(self.X_test, self.y_test)]
