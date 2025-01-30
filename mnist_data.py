#!/usr/bin/env python3
import os
import numpy as np
from dspy import Example
from typing import Tuple, List

class MNISTData:
    _dataset = None  # Class-level cache for loaded dataset

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """Initialize the MNIST data loader.

        Args:
            test_size (float, optional): Proportion of data for testing. Defaults to 0.2.
            random_state (int, optional): Seed for random operations. Defaults to 42.
        """
        self.test_size = test_size
        self.random_state = random_state
        self._load_data()

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess the MNIST dataset.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                (train_images, train_labels, test_images, test_labels)
        """
        # Load MNIST dataset
        train_images = np.random.rand(60000, 784)  # Simulated training images
        train_labels = np.random.randint(0, 10, 60000)  # Simulated training labels
        test_images = np.random.rand(10000, 784)  # Simulated test images
        test_labels = np.random.randint(0, 10, 10000)  # Simulated test labels

        # Apply random state for reproducibility
        rng = np.random.default_rng(self.random_state)
        rng.shuffle(train_images, axis=0)
        rng.shuffle(train_labels)
        
        # Split dataset
        train_idx, test_idx = self._split_indices(train_images.shape[0], self.test_size, self.random_state)
        train_images, train_labels = train_images[train_idx], train_labels[train_idx]
        test_images, test_labels = test_images[test_idx], test_labels[test_idx]

        return train_images, train_labels, test_images, test_labels

    def _matrix_to_text(self, matrix: np.ndarray) -> str:
        """Convert a matrix to a text representation.

        Args:
            matrix (np.ndarray): The matrix to convert.

        Returns:
            str: Text representation of the matrix.
        """
        return '\n'.join([''.join(['#' if pixel > 0.5 else ' ' for pixel in row]) for row in matrix])

    def get_test_data(self) -> List[Example]:
        """Get the test data as a list of dspy.Example objects.

        Returns:
            List[Example]: List of test examples.
        """
        train_images, train_labels, test_images, test_labels = self._load_data()
        
        examples = []
        for image, label in zip(test_images, test_labels):
            example = Example(
                inputs={"pixel_matrix": image},
                labels={"digit": str(label)}
            )
            examples.append(example)
        return examples

    def get_training_data(self) -> List[Example]:
        """Get the training data as a list of dspy.Example objects.

        Returns:
            List[Example]: List of training examples.
        """
        train_images, train_labels, _, _ = self._load_data()
        
        examples = []
        for image, label in zip(train_images, train_labels):
            example = Example(
                inputs={"pixel_matrix": image},
                labels={"digit": str(label)}
            )
            examples.append(example)
        return examples

    def _split_indices(self, total: int, test_size: float, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
        """Split indices into training and test sets.

        Args:
            total (int): Total number of samples.
            test_size (float): Proportion of samples for testing.
            random_state (int): Seed for random operations.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (train_indices, test_indices)
        """
        indices = np.arange(total)
        rng = np.random.default_rng(random_state)
        train_idx, test_idx = np.array_split(indices, [int(total * (1 - test_size))], axis=0)
        return train_idx, test_idx
