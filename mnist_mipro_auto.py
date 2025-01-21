#!/usr/bin/env python3
import argparse
import random
import dspy
from dspy.teleprompt import MIPROv2
from mnist_dspy import MNISTClassifier, MNISTBooster, create_training_data, create_test_data
from mnist_evaluation import MNISTEvaluator

class MNISTMIPROTrainer:
    """Train MNIST classifier using MIPROv2 optimization with configurable automation levels."""
    
    def __init__(self, auto_setting: str = "light", model_name: str = "deepseek/deepseek-chat", no_cache: bool = False, boosting_iterations: int = 1):
        # Initialize run configuration
        self.run_config = {
            'model': 'MNISTClassifier',
            'optimizer': f'MIPROv2 (auto={auto_setting})',
            'auto_setting': auto_setting,
            'train_samples': 5000,  # Match increased sample size
            'test_samples': 1000,
            'num_threads': 100,  # High parallelism for maximum throughput
            'random_state': 42,
            'model_name': model_name,
            'no_cache': no_cache,
            'boosting_iterations': boosting_iterations
        }
        
        self.classifier = MNISTBooster(
            model_name=model_name,
            boosting_iterations=boosting_iterations
        )
        # Create training data with proper dspy.Example format
        raw_train = create_training_data(samples=self.run_config['train_samples'])
        # Shuffle and sample training data
        random.shuffle(raw_train)
        sampled_train = raw_train[:self.run_config['train_samples']]
        self.train_data = [
            dspy.Example(pixel_matrix=pixels, digit=str(label)).with_inputs('pixel_matrix')
            for pixels, label in sampled_train
        ]
        
        # Create test data with proper dspy.Example format
        raw_test = create_test_data(samples=self.run_config['test_samples'])
        self.test_data = [
            dspy.Example(pixel_matrix=pixels, digit=str(label)).with_inputs('pixel_matrix')
            for pixels, label in raw_test
        ]
        self.evaluator = MNISTEvaluator(model_name=model_name, no_cache=no_cache)

    def _accuracy_metric(self, example, pred, trace=None):
        # Ensure both values are strings and compare
        true_label = str(example.digit) if hasattr(example, 'digit') else str(example)
        pred_label = str(pred.digit) if hasattr(pred, 'digit') else str(pred)
        return true_label == pred_label

    def train(self):
        # Evaluate baseline model before optimization
        print("Evaluating baseline model before optimization...")
        baseline_accuracy = self.evaluator.evaluate_accuracy(self.test_data)
        self.run_config['baseline_accuracy'] = float(baseline_accuracy)
        print(f"Baseline accuracy: {baseline_accuracy:.2%}")
        
        print(f"Initializing MIPROv2 with auto={self.run_config['auto_setting']}...")
        teleprompter = MIPROv2(
            metric=self._accuracy_metric,
            auto=self.run_config['auto_setting'],
            num_threads=self.run_config['num_threads']
        )
        
        print("Starting training...")
        print(f"Training on {len(self.train_data)} samples")
        self.optimized_classifier = teleprompter.compile(
            self.classifier,
            trainset=self.train_data,
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

def parse_args():
    parser = argparse.ArgumentParser(description='Train MNIST classifier with MIPROv2',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Optimization preset selection
    parser.add_argument('--auto', choices=['light', 'medium', 'heavy'], default='light',
                      help='Optimization level: light (fastest), medium (balanced), heavy (most thorough)')
    
    # Model selection
    parser.add_argument('--model', choices=['reasoner', 'chat'], default='chat',
                      help='Base model: reasoner (specialized) or chat (general purpose)')
    
    # Performance options
    parser.add_argument('--no-cache', action='store_true',
                      help='Disable model response caching (slower but fresh results)')
    parser.add_argument('--boosting', type=int, default=1,
                      help='Number of boosting iterations for ensemble voting')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Get auto setting from command line
    auto_setting = args.auto
        
    # Set model based on selection
    model_name = 'deepseek/deepseek-reasoner' if args.model == 'reasoner' else 'deepseek/deepseek-chat'
        
    print(f"Running MNIST Trainer with MIPROv2 (auto={auto_setting})")
    trainer = MNISTMIPROAutoTrainer(
        auto_setting=auto_setting,
        model_name=model_name,
        no_cache=args.no_cache,
        boosting_iterations=args.boosting
    )
    print("Training model...")
    trainer.train()
    accuracy = trainer.evaluate()
    print(f"Optimized model accuracy: {accuracy:.2%}")
    
    print("\n=== Final Run Configuration ===")
    for key, value in trainer.run_config.items():
        print(f"{key}: {value}")
    print("==============================")

if __name__ == "__main__":
    main()
