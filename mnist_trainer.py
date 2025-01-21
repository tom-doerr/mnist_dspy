#!/usr/bin/env python3
import dspy
from dspy.teleprompt import MIPROv2
from mnist_dspy import MNISTClassifier, create_training_data, create_test_data
from mnist_evaluation import MNISTEvaluator

class MNISTTrainer:
    def __init__(self):
        # Initialize run configuration
        self.run_config = {
            'model': 'MNISTClassifier',
            'optimizer': 'MIPROv2',
            'max_bootstrapped_demos': 20,
            'max_labeled_demos': 20,
            'num_threads': 100,  # High parallelism for maximum throughput
            'train_samples': 5000,  # Increased sample size
            'test_samples': 1000,
            'requires_permission': False,
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
        self.evaluator = MNISTEvaluator(classifier=self.classifier)

    def _accuracy_metric(self, example, pred, trace=None):
        # Ensure both values are strings and compare
        true_label = str(example.digit) if hasattr(example, 'digit') else str(example)
        pred_label = str(pred.digit) if hasattr(pred, 'digit') else str(pred)
        return true_label == pred_label

    def train(self, data):
        # Evaluate baseline model before optimization
        print("Evaluating baseline model before optimization...")
        baseline_accuracy = self.evaluator.evaluate_accuracy(self.test_data)
        self.run_config['baseline_accuracy'] = float(baseline_accuracy)
        print(f"Baseline accuracy: {baseline_accuracy:.2%}")
        
        print("Initializing MIPROv2 optimizer...")
        teleprompter = MIPROv2(
            metric=self._accuracy_metric,
            max_bootstrapped_demos=self.run_config['max_bootstrapped_demos'],
            max_labeled_demos=self.run_config['max_labeled_demos'],
            num_threads=self.run_config['num_threads']
        )
        
        print("Starting training with MIPROv2...")
        print(f"Training on {len(data)} samples")
        self.optimized_classifier = teleprompter.compile(
            self.classifier,
            trainset=data,
            requires_permission_to_run=False
        )
        print("Training completed successfully")
        
        return self.optimized_classifier

    def evaluate(self):
        if not hasattr(self, 'optimized_classifier'):
            raise ValueError("Model must be trained before evaluation")
            
        print("Evaluating model on test data...")
        print(f"Using {len(self.test_data)} test samples")
        # Create new evaluator with optimized model
        optimized_evaluator = MNISTEvaluator()
        optimized_evaluator.inference.classifier = self.optimized_classifier
        accuracy = optimized_evaluator.evaluate_accuracy(self.test_data)
        
        # Add final results to run config
        self.run_config['final_accuracy'] = float(accuracy)
        
        print("\n=== Run Configuration ===")
        for key, value in self.run_config.items():
            print(f"{key}: {value}")
        print("========================")
        
        print("\nEvaluation completed")
        return accuracy

if __name__ == "__main__":
    print("Running MNIST Trainer with MIPROv2 optimizer")
    trainer = MNISTTrainer()
    print("Training model...")
    trainer.train()
    accuracy = trainer.evaluate()
    print(f"Optimized model accuracy: {accuracy:.2%}")
    
    print("\n=== Final Run Configuration ===")
    for key, value in trainer.run_config.items():
        print(f"{key}: {value}")
    print("==============================")
