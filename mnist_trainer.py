#!/usr/bin/env python3
import dspy
from dspy.teleprompt import MIPROv2
from mnist_dspy import MNISTClassifier
from mnist_data import MNISTData
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class MNISTTrainer:
    DEFAULT_NUM_WORKERS = 10  # Static variable for default number of workers
    DEFAULT_MODEL_NAME = "deepseek/deepseek-chat"  # Static variable for default model name

    def __init__(self, optimizer: str = "MIPROv2", iterations: int = 1,
                 model_name: str = DEFAULT_MODEL_NAME, auto: str = "light"):
        """Initialize the MNIST trainer with specified parameters.

        Args:
            optimizer (str, optional): The optimizer to use. Defaults to "MIPROv2".
            iterations (int, optional): Number of optimization iterations. Defaults to 1.
            model_name (str, optional): The model to use. Defaults to DEFAULT_MODEL_NAME.
            auto (str, optional): Auto optimization setting for MIPROv2. Defaults to "light".
        """
        self.optimizer = optimizer
        self.model_name = model_name
        self.iterations = iterations
        self.auto = auto
        print(f"\nInitializing trainer with:")
        print(f"- Optimizer: {optimizer}")
        print(f"- Model: {self.model_name}")
        print(f"- Auto setting: {auto}")
        print(f"- Iterations: {iterations}\n")
        
        self.classifier = MNISTClassifier(model_name=self.model_name)
        mnist_data = MNISTData()
        with tqdm(desc="Loading training data") as pbar:
            self.train_data = mnist_data.get_training_data()[:10000]  # Get 1000 training samples
            pbar.update(1)
        with tqdm(desc="Loading test data") as pbar:
            self.test_data = mnist_data.get_test_data()[:200]  # Get 200 test samples
            pbar.update(1)

    def _accuracy_metric(self, example, pred, trace=None):
        """Calculate the accuracy metric by comparing the predicted digit with the actual digit.

        Args:
            example: The example containing the actual digit.
            pred: The prediction containing the predicted digit.
            trace (optional): The trace of the prediction process.

        Returns:
            bool: True if the prediction matches the actual digit, False otherwise.
        """
        return str(example.digit) == str(pred.digit)

    def train(self, data):
        """Train the model using the specified data.

        Args:
            data: The training data to use.

        Returns:
            The optimized classifier.
        """
        print("Evaluating baseline model before optimization...")
        correct = 0
        total = len(self.test_data)
        with tqdm(total=total, desc="Evaluating baseline") as pbar:
            for example in self.test_data:
                pred = self.classifier(example.pixel_matrix)
                if str(pred.digit) == str(example.digit):
                    correct += 1
                pbar.update(1)
        baseline_accuracy = correct / total
        print(f"Baseline accuracy: {baseline_accuracy:.2%}")
        self.baseline_accuracy = baseline_accuracy
        
        optimizer_class = MIPROv2 if self.optimizer == 'MIPROv2' else dspy.teleprompt.BootstrapFewShot
        if self.optimizer == 'MIPROv2':
            teleprompter = optimizer_class(
                metric=self._accuracy_metric,
                num_threads=self.DEFAULT_NUM_WORKERS,
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
        """Evaluate the optimized model on the test data.

        Returns:
            float: The accuracy of the model on the test data.
        """
        if not hasattr(self, 'optimized_classifier'):
            raise ValueError("Model must be trained before evaluation")
            
        print("Evaluating model on test data...")
        print(f"Using {len(self.test_data)} test samples with {self.DEFAULT_NUM_WORKERS} threads")
        
        with tqdm(total=len(self.test_data), desc="Evaluating") as pbar:
            correct = 0
            for example in self.test_data:
                pred = self.optimized_classifier(example.pixel_matrix)
                if str(pred.digit) == str(example.digit):
                    correct += 1
                current_accuracy = correct / len(self.test_data)
                pbar.set_postfix({'accuracy': f"{current_accuracy:.2%}"})
                pbar.update(1)
                
        accuracy = correct / len(self.test_data)
        
        print(f"\n\nBaseline accuracy: {self.baseline_accuracy:.2%}")
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
    parser.add_argument('--model', default=None, help='Model type to use')
    parser.add_argument('--auto', choices=['light', 'medium', 'heavy'],
                      default='light', help='Auto optimization setting for MIPROv2')
    args = parser.parse_args()
    
    model_name = args.model or MNISTTrainer.DEFAULT_MODEL_NAME
    
    print(f"Running MNIST Trainer with {args.optimizer}")
    trainer = MNISTTrainer(
        optimizer=args.optimizer,
        iterations=args.iterations,
        model_name=model_name,
        auto=args.auto
    )
    
    print("Training model...")
    with tqdm(ncols=75, desc="Training progress") as pbar:
        pbar.set_description("Training model")
        trainer.train(trainer.train_data)
        pbar.update(1)
    
    print("\nEvaluating optimized model on test data...")
    with tqdm(ncols=75, desc="Evaluation progress") as pbar:
        pbar.set_description("Evaluating model")
        accuracy = trainer.evaluate()
        pbar.update(1)
    print(f"Final test accuracy: {accuracy:.2%}")
