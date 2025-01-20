from typing import List, Tuple
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split

class MNISTData:
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = self._load_data()

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        print("Loading MNIST data from Hugging Face...")
        dataset = load_dataset("mnist")
        
        # Convert to numpy arrays
        X_train = np.array(dataset['train']['image'])
        y_train = np.array(dataset['train']['label'])
        X_test = np.array(dataset['test']['image'])
        y_test = np.array(dataset['test']['label'])
        
        # Convert images to 784-dimensional vectors
        X_train = np.array([np.array(img).flatten() for img in X_train])
        X_test = np.array([np.array(img).flatten() for img in X_test])
        
        print(f"Loaded {len(X_train)} training samples")
        print(f"Loaded {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test

    def _matrix_to_text(self, matrix: np.ndarray) -> str:
        reshaped = matrix.reshape(28, 28)
        return '\n'.join(' '.join(str(pixel) for pixel in row) for row in reshaped)

    def get_training_data(self) -> List[Tuple[str, int]]:
        return [(self._matrix_to_text(x), int(y)) for x, y in zip(self.X_train, self.y_train)]

    def get_test_data(self) -> List[Tuple[str, int]]:
        return [(self._matrix_to_text(x), int(y)) for x, y in zip(self.X_test, self.y_test)]
