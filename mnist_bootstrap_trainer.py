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
            'max_bootstrapped_demos': 10,
            'max_labeled_demos': 10,
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

    def train(self) -> MNISTClassifier:
        # Evaluate baseline model before optimization
        print("Evaluating baseline model before optimization...")
        baseline_accuracy = self.evaluator.evaluate_accuracy(self.test_data)
        self.run_config['baseline_accuracy'] = float(baseline_accuracy)
        print(f"Baseline accuracy: {baseline_accuracy:.2%}")
        
        print("Initializing BootstrapFewShot optimizer...")
        teleprompter = BootstrapFewShot(
            metric=self._accuracy_metric,
            max_bootstrapped_demos=5,  # Few-shot demos from bootstrap
            max_labeled_demos=3,       # Small set of labeled examples
            max_rounds=2,              # Limited optimization passes
            max_errors=3               # Tolerance for few errors
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

def parse_args():
    parser = argparse.ArgumentParser(description='Train MNIST classifier with BootstrapFewShot')
    parser.add_argument('--model', choices=['reasoner', 'chat'], default='chat',
                      help='Base model: reasoner (specialized) or chat (general purpose)')
    parser.add_argument('--train-samples', type=int, default=1000,
                      help='Number of training samples to use')
    parser.add_argument('--verbose', action='store_true',
                      help='Show detailed prediction outputs')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set model based on selection
    model_name = 'deepseek/deepseek-reasoner' if args.model == 'reasoner' else 'deepseek/deepseek-chat'
    
    print(f"Running MNIST Trainer with BootstrapFewShot optimizer ({args.model} model)")
    trainer = MNISTBootstrapTrainer(
        model_name=model_name
    )
    trainer.run_config['train_samples'] = args.train_samples
    trainer.classifier.verbose = args.verbose
    
    print("Training model...")
    trainer.train()
    accuracy = trainer.evaluate()
    print(f"\nOptimized model accuracy: {accuracy:.2%}")
