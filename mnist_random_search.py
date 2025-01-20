#!/usr/bin/env python3
import argparse
import random
from typing import Dict, List
import dspy
from dspy.teleprompt import BootstrapFewShot
from mnist_dspy import MNISTClassifier, create_training_data, create_test_data
from mnist_evaluation import MNISTEvaluator

class MNISTRandomSearch:
    def __init__(self, max_trials: int = 10):
        self.max_trials = max_trials
        self.results = []
        self.best_config = None
        self.best_accuracy = 0.0

    def _generate_random_config(self) -> Dict:
        return {
            'num_bootstrapped_demos': random.randint(10, 50),
            'max_labeled_demos': random.randint(10, 50),
            'train_samples': random.randint(1000, 5000),
            'temperature': random.uniform(0.1, 2.0),
            'max_answer_length': 2 ** random.randint(6, 9),  # 64, 128, 256, 512
            'model': 'chat'
        }

    def run_search(self):
        for trial in range(self.max_trials):
            config = self._generate_random_config()
            print(f"\n=== Trial {trial+1}/{self.max_trials} ===")
            print("Testing config:", config)
            
            try:
                # Initialize components with current config
                classifier = MNISTClassifier()
                evaluator = MNISTEvaluator()
                
                # Create training data
                raw_train = create_training_data()
                # Shuffle and sample training data
                random.shuffle(raw_train)
                sampled_train = raw_train[:config['train_samples']]
                train_data = [
                    dspy.Example(pixel_matrix=pixels, digit=str(label)).with_inputs('pixel_matrix')
                    for pixels, label in sampled_train
                ]
                
                # Configure optimizer
                teleprompter = BootstrapFewShot(
                    max_bootstrapped_demos=config['num_bootstrapped_demos'],
                    max_labeled_demos=config['max_labeled_demos']
                )
                
                # Train and evaluate
                optimized_classifier = teleprompter.compile(classifier, trainset=train_data)
                # Create new evaluator with optimized model
                optimized_evaluator = MNISTEvaluator()
                optimized_evaluator.inference.classifier = optimized_classifier
                accuracy = optimized_evaluator.evaluate_accuracy(create_test_data())
                
                # Track results
                result = config.copy()
                result['accuracy'] = accuracy
                self.results.append(result)
                
                # Update best config
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_config = config
                
                print(f"Trial accuracy: {accuracy:.2%}")
                print(f"   Best so far: {self.best_accuracy:.2%} with")
                print("   " + "\n   ".join(f"{k}: {v}" for k,v in self.best_config.items()))
                
            except Exception as e:
                print(f"⚠️ Config failed: {str(e)}")
                self.results.append(config | {'error': str(e), 'accuracy': 0.0})
        
        return self.best_config

    def print_results(self):
        print("\n=== Random Search Results ===")
        for i, result in enumerate(self.results):
            status = f"{result['accuracy']:.2%}" if 'accuracy' in result else f"Error: {result.get('error', 'Unknown')}"
            print(f"Trial {i+1}: {status}")
        
        print("\n⭐ Best Configuration:")
        for k, v in self.best_config.items():
            print(f"{k}: {v}")
        print(f"Best Accuracy: {self.best_accuracy:.2%}")

def main():
    parser = argparse.ArgumentParser(description='Run random hyperparameter search for MNIST classification')
    parser.add_argument('--trials', type=int, default=10,
                      help='Number of random configurations to test')
    args = parser.parse_args()
    
    searcher = MNISTRandomSearch(max_trials=args.trials)
    searcher.run_search()
    searcher.print_results()

if __name__ == "__main__":
    main()
