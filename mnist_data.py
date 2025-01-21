#!/usr/bin/env python3
from typing import List, Tuple
import numpy as np
import dspy
import random
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


    def get_test_data(self) -> List[dspy.Example]:
        test_data = [
            dspy.Example(pixel_matrix=self._matrix_to_text(x), digit=int(y)).with_inputs("pixel_matrix")  # Creating base test examples
            for x, y in zip(self.X_test, self.y_test)
        ]
        return test_data

    def get_training_data(self) -> List[dspy.Example]:
        # Shuffle training data to ensure random sample
        shuffled = list(zip(self.X_train, self.y_train))
        random.shuffle(shuffled)
        X_shuffled, y_shuffled = zip(*shuffled)
        
        train_data = [
            dspy.Example(pixel_matrix=self._matrix_to_text(x), digit=int(y)).with_inputs("pixel_matrix")
            for x, y in zip(X_shuffled, y_shuffled)
        ]
        return train_data
