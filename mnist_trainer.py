#!/usr/bin/env python3
import dspy
from dspy.teleprompt import MIPROv2
from mnist_dspy import MNISTClassifier
from mnist_data import MNISTData

class MNISTTrainer:
    def __init__(self, optimizer: str = "MIPROv2", iterations: int = 1,
                 model_name: str = "deepseek/deepseek-chat", auto: str = "light"):
        self.optimizer = optimizer
        self.model_name = model_name
        self.iterations = iterations
        self.auto = auto
        print(f"\nInitializing trainer with:")
        print(f"- Optimizer: {optimizer}")
        print(f"- Model: {model_name}")
        print(f"- Auto setting: {auto}")
        print(f"- Iterations: {iterations}\n")
        
        self.classifier = MNISTClassifier(model_name=model_name)
        mnist_data = MNISTData()
        self.train_data = mnist_data.get_training_data()[:1000]  # Get 1000 training samples
        self.test_data = mnist_data.get_test_data()[:200]  # Get 200 test samples

    def _accuracy_metric(self, example, pred, trace=None):
        return str(example.digit) == str(pred.digit)

    def train(self, data):
        print("Evaluating baseline model before optimization...")
        correct = 0
        total = len(self.test_data)
        for example in self.test_data:
            pred = self.classifier(example.pixel_matrix)
            if str(pred.digit) == str(example.digit):
                correct += 1
        baseline_accuracy = correct / total
        print(f"Baseline accuracy: {baseline_accuracy:.2%}")
        
        optimizer_class = MIPROv2 if self.optimizer == 'MIPROv2' else dspy.teleprompt.BootstrapFewShot
        if self.optimizer == 'MIPROv2':
            teleprompter = optimizer_class(
                metric=self._accuracy_metric,
                auto=self.auto
            )
        else:
            teleprompter = optimizer_class(
                metric=self._accuracy_metric,
                max_bootstrapped_demos=10,
                max_labeled_demos=10
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
        correct = 0
        total = len(self.test_data)
        for example in self.test_data:
            pred = self.optimized_classifier(example.pixel_matrix)
            if str(pred.digit) == str(example.digit):
                correct += 1
        accuracy = correct / total
        
        print(f"\nOptimizer: {self.optimizer}")
        print(f"Model: {self.model_name}")
        print(f"Iterations: {self.iterations}")
        print(f"Final accuracy: {accuracy:.2%}")
        
        return accuracy

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MNIST classifier')
    parser.add_argument('--optimizer', choices=['MIPROv2', 'BootstrapFewShot'], 
                      default='MIPROv2', help='Optimizer to use')
    parser.add_argument('--iterations', type=int, default=1,
                      help='Number of optimization iterations')
    parser.add_argument('--model', choices=['reasoner', 'chat'],
                      default='chat', help='Model type to use')
    parser.add_argument('--auto', choices=['light', 'medium', 'heavy'],
                      default='light', help='Auto optimization setting for MIPROv2')
    args = parser.parse_args()
    
    model_name = 'deepseek/deepseek-reasoner' if args.model == 'reasoner' else 'deepseek/deepseek-chat'
    
    print(f"Running MNIST Trainer with {args.optimizer}")
    trainer = MNISTTrainer(
        optimizer=args.optimizer,
        iterations=args.iterations,
        model_name=model_name,
        auto=args.auto
    )
    
    print("Training model...")
    trainer.train(trainer.train_data)
    accuracy = trainer.evaluate()
    print(f"\nOptimized model accuracy: {accuracy:.2%}")
