#!/usr/bin/env python3
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
        # Optimized string conversion using numpy operations
        return '\n'.join([' '.join(map(str, row)) for row in reshaped])

    def get_training_data(self, validation_ratio: float = 0.1) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """Split training data into train/validation sets"""
        full_data = [(self._matrix_to_text(x), int(y)) for x, y in zip(self.X_train, self.y_train)]
        val_size = int(len(full_data) * validation_ratio)
        return full_data[val_size:], full_data[:val_size]

    def get_test_data(self) -> List[Tuple[str, int]]:
        test_data = [(self._matrix_to_text(x), int(y)) for x, y in zip(self.X_test, self.y_test)]
        print(f"\nData Loader Debug:")
        print(f"Total test samples: {len(test_data)}")
        print(f"Sample label: {test_data[0][1]}")
        print(f"Sample pixels start: {test_data[0][0][:50]}...")
        return test_data
