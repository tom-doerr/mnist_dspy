#!/usr/bin/env python3
import dspy
from dspy.teleprompt import MIPROv2
from mnist_dspy import MNISTClassifier, create_training_data, create_test_data
from mnist_evaluation import MNISTEvaluator

class MNISTTrainer:
    def __init__(self, optimizer: str = "MIPROv2", auto_setting: str = "light", 
                 bootstrap_iterations: int = 1, model_name: str = "deepseek/deepseek-chat"):
        self.run_config = {
            'model': model_name,
            'optimizer': optimizer,
            'auto_setting': auto_setting,
            'bootstrap_iterations': bootstrap_iterations,
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
            dspy.Example(pixel_matrix=pixels, number=str(label)).with_inputs('pixel_matrix')
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

    def train(self, data):
        print("Evaluating baseline model before optimization...")
        baseline_accuracy = self.evaluator.evaluate_accuracy(self.test_data, predictor=self.classifier)
        self.run_config['baseline_accuracy'] = float(baseline_accuracy)
        print(f"Baseline accuracy: {baseline_accuracy:.2%}")
        
        if self.run_config['optimizer'] == 'MIPROv2':
            print("Initializing MIPROv2 optimizer...")
            teleprompter = MIPROv2(
                metric=self._accuracy_metric,
                max_bootstrapped_demos=self.run_config['max_bootstrapped_demos'],
                max_labeled_demos=self.run_config['max_labeled_demos'],
                num_threads=self.run_config['num_threads']
            )
        else:  # BootstrapFewShot
            print("Initializing BootstrapFewShot optimizer...")
            teleprompter = dspy.teleprompt.BootstrapFewShot(
                metric=self._accuracy_metric,
                max_bootstrapped_demos=self.run_config['max_bootstrapped_demos'],
                max_labeled_demos=self.run_config['max_labeled_demos']
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MNIST classifier')
    parser.add_argument('--optimizer', choices=['MIPROv2', 'BootstrapFewShot'], 
                      default='MIPROv2', help='Optimizer to use')
    parser.add_argument('--auto', choices=['light', 'medium', 'heavy'],
                      default='light', help='Optimization level')
    parser.add_argument('--iterations', type=int, default=1,
                      help='Number of bootstrap iterations')
    parser.add_argument('--model', choices=['reasoner', 'chat'],
                      default='chat', help='Model type to use')
    args = parser.parse_args()
    
    model_name = 'deepseek/deepseek-reasoner' if args.model == 'reasoner' else 'deepseek/deepseek-chat'
    
    print(f"Running MNIST Trainer with {args.optimizer}")
    trainer = MNISTTrainer(
        optimizer=args.optimizer,
        auto_setting=args.auto,
        bootstrap_iterations=args.iterations,
        model_name=model_name
    )
    
    print("Training model...")
    trainer.train(trainer.train_data)
    accuracy = trainer.evaluate()
    print(f"\nOptimized model accuracy: {accuracy:.2%}")
