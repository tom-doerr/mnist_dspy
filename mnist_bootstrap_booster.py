#!/usr/bin/env python3
import random
import dspy
from typing import List
from dspy.teleprompt import BootstrapFewShot
from mnist_dspy import MNISTClassifier, create_training_data
from mnist_evaluation import MNISTEvaluator

class MNISTBootstrapBooster:
    def __init__(self, num_models: int = 3, model_name: str = "deepseek/deepseek-chat"):
        self.num_models = num_models
        self.model_name = model_name
        self.models: List[MNISTClassifier] = []
        self.optimizers: List[BootstrapFewShot] = []
        
        # Initialize base components
        self.evaluator = MNISTEvaluator(model_name=model_name)
        self.raw_train = create_training_data()

    def train(self) -> None:
        """Train ensemble of models with bootstrap sampling"""
        for i in range(self.num_models):
            print(f"\n=== Training model {i+1}/{self.num_models} ===")
            
            # Create new classifier and optimizer
            model = MNISTClassifier(model_name=self.model_name)
            optimizer = BootstrapFewShot(
                max_bootstrapped_demos=20,
                max_labeled_demos=20
            )
            
            # Shuffle and sample training data with replacement
            random.shuffle(self.raw_train)
            train_data = [
                dspy.Example(pixel_matrix=pixels, digit=str(label)).with_inputs('pixel_matrix')
                for pixels, label in self.raw_train
            ]
            
            # Train model
            optimized_model = optimizer.compile(model, trainset=train_data)
            self.models.append(optimized_model)
            self.optimizers.append(optimizer)

    def evaluate(self, test_data: List[dspy.Example]) -> float:
        """Evaluate ensemble using majority voting"""
        correct = 0
        for example in test_data:
            predictions = [model(pixel_matrix=example.pixel_matrix).digit 
                          for model in self.models]
            majority_vote = max(set(predictions), key=predictions.count)
            if majority_vote == example.digit:
                correct += 1
        return correct / len(test_data)

    def forward(self, pixel_matrix: str) -> str:
        """Make prediction with ensemble voting"""
        predictions = [model(pixel_matrix=pixel_matrix).digit 
                      for model in self.models]
        return max(set(predictions), key=predictions.count)

if __name__ == "__main__":
    # Example usage
    booster = MNISTBootstrapBooster(num_models=3)
    print("Starting training...")
    booster.train()
    
    # Evaluate on subset of test data
    test_data = create_training_data()[100:150]  # Use validation set for quick test
    accuracy = booster.evaluate(test_data)
    print(f"\nEnsemble accuracy: {accuracy:.2%}")
