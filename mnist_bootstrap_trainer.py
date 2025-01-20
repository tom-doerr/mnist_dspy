#!/usr/bin/env python3
import dspy
from dspy.teleprompt import BootstrapFewShot
from mnist_dspy import MNISTClassifier, create_training_data, create_test_data
from mnist_evaluation import MNISTEvaluator

class MNISTBootstrapTrainer:
    def __init__(self):
        # Initialize run configuration
        self.run_config = {
            'model': 'MNISTClassifier',
            'optimizer': 'BootstrapFewShot',
            'max_bootstrapped_demos': 100,
            'max_labeled_demos': 100,
            'num_threads': 100,
            'train_samples': 1000,
            'test_samples': 200,
            'random_state': 42
        }
        
        self.classifier = MNISTClassifier()
        # Create training data with proper dspy.Example format
        raw_train = create_training_data()
        self.train_data = [
            dspy.Example(pixel_matrix=pixels, digit=str(label)).with_inputs('pixel_matrix')
            for pixels, label in raw_train
        ]
        
        # Create test data with proper dspy.Example format
        raw_test = create_test_data()
        self.test_data = [
            dspy.Example(pixel_matrix=pixels, digit=str(label)).with_inputs('pixel_matrix')
            for pixels, label in raw_test
        ]
        self.evaluator = MNISTEvaluator()

    def _accuracy_metric(self, example, pred, trace=None):
        # Ensure both values are strings and compare
        true_label = str(example.digit) if hasattr(example, 'digit') else str(example)
        pred_label = str(pred.digit) if hasattr(pred, 'digit') else str(pred)
        return true_label == pred_label

    def train(self):
        print("Initializing BootstrapFewShot optimizer...")
        teleprompter = BootstrapFewShot(
            metric=self._accuracy_metric,
            max_bootstrapped_demos=self.run_config['max_bootstrapped_demos'],
            max_labeled_demos=self.run_config['max_labeled_demos']
        )
        
        print("Starting training with BootstrapFewShot...")
        print(f"Training on {len(self.train_data)} samples")
        self.optimized_classifier = teleprompter.compile(
            self.classifier,
            trainset=self.train_data
        )
        print("Training completed successfully")
        
        return self.optimized_classifier

    def evaluate(self):
        if not hasattr(self, 'optimized_classifier'):
            raise ValueError("Model must be trained before evaluation")
            
        print("Evaluating model on test data...")
        print(f"Using {len(self.test_data)} test samples")
        accuracy = self.evaluator.evaluate_accuracy(self.test_data)
        
        # Add final results to run config
        self.run_config['final_accuracy'] = float(accuracy)
        
        print("\n=== Run Configuration ===")
        for key, value in self.run_config.items():
            print(f"{key}: {value}")
        print("========================")
        
        print("\nEvaluation completed")
        return accuracy

if __name__ == "__main__":
    print("Running MNIST Trainer with BootstrapFewShot optimizer")
    trainer = MNISTBootstrapTrainer()
    print("Training model...")
    trainer.train()
    accuracy = trainer.evaluate()
    print(f"Optimized model accuracy: {accuracy:.2%}")
