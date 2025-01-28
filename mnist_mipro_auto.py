#!/usr/bin/env python3
import argparse
import dspy
from dspy.teleprompt import MIPROv2, BootstrapFewShot
from mnist_dspy import MNISTClassifier, create_training_data, create_test_data
from mnist_evaluation import MNISTEvaluator
class MNISTTrainer:
    """Train MNIST classifier using configurable optimization strategies."""
    
    def __init__(self, optimizer: str = "MIPROv2", auto: str = "light", iterations: int = 3):
        self.config = {
            'optimizer': optimizer,
            'auto': auto,
            'iterations': iterations,
            'train_samples': 1000,
            'test_samples': 200,
            'num_threads': 8
        }
        
        # Initialize base classifier
        self.classifier = MNISTClassifier()
        # Create datasets
        self.train_data = create_training_data(samples=self.config['train_samples'])
        self.test_data = create_test_data(samples=self.config['test_samples'])
        self.evaluator = MNISTEvaluator()

    def train(self):
        print(f"\nInitializing {self.config['optimizer']} optimizer...")
        
        if self.config['optimizer'] == 'MIPROv2':
            optimizer = MIPROv2(
                metric=dspy.evaluate.answer_exact_match,
                auto=self.config['auto'],
                num_threads=self.config['num_threads']
            )
        else:
            optimizer = BootstrapFewShot(
                metric=dspy.evaluate.answer_exact_match,
                max_bootstrapped_demos=self.config['iterations'],
                num_threads=self.config['num_threads']
            )
        
        print(f"Training on {len(self.train_data)} samples...")
        self.optimized_classifier = optimizer.compile(
            self.classifier,
            trainset=self.train_data
        )
        
        print("\nEvaluating on test data...")
        accuracy = self.evaluator.evaluate_accuracy(self.test_data)
        
        print(f"\nFinal accuracy: {accuracy:.2%}")
        return accuracy

def main():
    parser = argparse.ArgumentParser(description='Train MNIST classifier')
    parser.add_argument('--optimizer', choices=['MIPROv2', 'BootstrapFewShot'], 
                      default='MIPROv2', help='Optimization strategy')
    parser.add_argument('--auto', choices=['light', 'medium', 'heavy'],
                      default='light', help='Optimization level for MIPROv2')
    parser.add_argument('--iterations', type=int, default=3,
                      help='Number of optimization iterations')
    args = parser.parse_args()
    
    trainer = MNISTTrainer(
        optimizer=args.optimizer,
        auto=args.auto,
        iterations=args.iterations
    )
    trainer.train()

if __name__ == "__main__":
    main()
