#!/usr/bin/env python3
import dspy
from dspy.teleprompt import MIPROv2
from mnist_dspy import MNISTClassifier, create_training_data, create_test_data
from mnist_evaluation import MNISTEvaluator

class MNISTTrainer:
    def __init__(self, optimizer: str = "MIPROv2", iterations: int = 1,
                 model_name: str = "deepseek/deepseek-chat"):
        self.optimizer = optimizer
        self.model_name = model_name
        self.iterations = iterations
        
        self.classifier = MNISTClassifier()
        raw_train = create_training_data()
        self.train_data = [
            dspy.Example(pixel_matrix=pixels, digit=str(label)).with_inputs('pixel_matrix')
            for pixels, label in raw_train
        ]
        
        raw_test = create_test_data()
        self.test_data = [
            dspy.Example(pixel_matrix=pixels, digit=str(label)).with_inputs('pixel_matrix')
            for pixels, label in raw_test
        ]
        self.evaluator = MNISTEvaluator()

    def _accuracy_metric(self, example, pred, trace=None):
        return str(example.digit) == str(pred.digit)

    def train(self, data):
        print("Evaluating baseline model before optimization...")
        baseline_accuracy = self.evaluator.evaluate_accuracy(self.test_data, predictor=self.classifier)
        print(f"Baseline accuracy: {baseline_accuracy:.2%}")
        
        optimizer_class = MIPROv2 if self.optimizer == 'MIPROv2' else dspy.teleprompt.BootstrapFewShot
        teleprompter = optimizer_class(
            metric=self._accuracy_metric,
            max_bootstrapped_demos=10,
            max_labeled_demos=10,
            **(dict(num_threads=100) if self.optimizer == 'MIPROv2' else {})
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
        optimized_evaluator = MNISTEvaluator()
        optimized_evaluator.inference.classifier = self.optimized_classifier
        accuracy = optimized_evaluator.evaluate_accuracy(self.test_data)
        
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
    args = parser.parse_args()
    
    model_name = 'deepseek/deepseek-reasoner' if args.model == 'reasoner' else 'deepseek/deepseek-chat'
    
    print(f"Running MNIST Trainer with {args.optimizer}")
    trainer = MNISTTrainer(
        optimizer=args.optimizer,
        iterations=args.iterations,
        model_name=model_name
    )
    
    print("Training model...")
    trainer.train(trainer.train_data)
    accuracy = trainer.evaluate()
    print(f"\nOptimized model accuracy: {accuracy:.2%}")
