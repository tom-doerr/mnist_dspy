#!/usr/bin/env python3
import argparse
import dspy
from dspy.teleprompt import MIPROv2
from mnist_dspy import MNISTClassifier, create_training_data, create_test_data
from mnist_evaluation import MNISTEvaluator

class MNISTMIPROAutoTrainer:
    def __init__(self, auto_setting: str = "light", model_name: str = "deepseek/deepseek-chat"):
        # Initialize run configuration
        self.run_config = {
            'model': 'MNISTClassifier',
            'optimizer': f'MIPROv2 (auto={auto_setting})',
            'auto_setting': auto_setting,
            'train_samples': 1000,
            'test_samples': 200,
            'random_state': 42,
            'model_name': model_name
        }
        
        self.classifier = MNISTClassifier(model_name=model_name)
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
        print(f"Initializing MIPROv2 with auto={self.run_config['auto_setting']}...")
        teleprompter = MIPROv2(
            metric=self._accuracy_metric,
            auto=self.run_config['auto_setting']
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
    parser = argparse.ArgumentParser(description='Train MNIST classifier with MIPROv2')
    parser.add_argument('--light', action='store_true', help='Use light optimization preset')
    parser.add_argument('--medium', action='store_true', help='Use medium optimization preset')
    parser.add_argument('--heavy', action='store_true', help='Use heavy optimization preset')
    parser.add_argument('--model', choices=['reasoner', 'chat'], default='chat',
                      help='Model to use: reasoner or chat (default: chat)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Determine auto setting based on args
    if args.medium:
        auto_setting = "medium"
    elif args.heavy:
        auto_setting = "heavy"
    else:
        auto_setting = "light"  # Default to light if no option specified
        
    # Set model based on selection
    model_name = 'deepseek/deepseek-reasoner' if args.model == 'reasoner' else 'deepseek/deepseek-chat'
        
    print(f"Running MNIST Trainer with MIPROv2 (auto={auto_setting})")
    trainer = MNISTMIPROAutoTrainer(auto_setting=auto_setting)
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
