#!/usr/bin/env python3
import os
import numpy as np
from dspy import Example
from typing import Tuple, List
from datasets import load_dataset

class MNISTData:
    _dataset = None  # Class-level cache for loaded dataset

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """Initialize the MNIST data loader.

        Args:
            test_size (float, optional): Proportion of data for testing. Defaults to 0.2.
            random_state (int, optional): Seed for random operations; `None` for system random.
        """
        self.test_size = test_size
        self.random_state = random_state
        self._load_data()

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess the MNIST dataset."""
        # Load MNIST dataset from HuggingFace
        dataset = load_dataset("mnist")
        train_ds = dataset["train"]
        test_ds = dataset["test"]
        
        # Extract images and labels
        train_images = np.array([example['image'] for example in train_ds])
        train_labels = np.array([example['label'] for example in train_ds])
        test_images = np.array([example['image'] for example in test_ds])
        test_labels = np.array([example['label'] for example in test_ds])
        
        # Convert images to numpy arrays and normalize
        train_images = np.array([np.array(img) for img in train_images]).astype('float32') / 255.0
        test_images = np.array([np.array(img) for img in test_images]).astype('float32') / 255.0
        
        # Reshape images to 784-dimensional vectors
        train_images = train_images.reshape((-1, 784))
        test_images = test_images.reshape((-1, 784))
        
        # Apply random state for reproducibility if specified
        if self.random_state is not None:
            rng = np.random.default_rng(self.random_state)
            train_idx = rng.permutation(len(train_images))
            train_images = train_images[train_idx]
            train_labels = train_labels[train_idx]
        
        return train_images, train_labels, test_images, test_labels


    def get_test_data(self) -> List[Example]:
        """Get the test data as a list of dspy.Example objects.

        Returns:
            List[Example]: List of test examples.
        """
        train_images, train_labels, test_images, test_labels = self._load_data()
        
        examples = []
        for image, label in zip(test_images, test_labels):
            example = Example(
                pixel_matrix=image,
                digit=str(label)
            ).with_inputs('pixel_matrix')
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
                pixel_matrix=image,
                digit=str(label)
            ).with_inputs('pixel_matrix')
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
