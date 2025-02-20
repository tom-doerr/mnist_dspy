#!/usr/bin/env python3
import dspy
from dspy.teleprompt import MIPROv2
from mnist_dspy import MNISTClassifier
from mnist_data import MNISTData
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
import random

class MNISTTrainer:
    def __init__(self, optimizer: str = "MIPROv2", iterations: int = 1,
                 model_name: str = "deepseek/deepseek-chat", auto: str = "light",
                 num_workers: int = 100, cache: bool = True):
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
        self.num_workers = num_workers
        self.cache = cache
        print(f"\nInitializing trainer with:")
        print(f"- Optimizer: {optimizer}")
        print(f"- Model: {self.model_name}")
        print(f"- Auto setting: {auto}")
        print(f"- Cache enabled: {cache}")
        print(f"- Iterations: {iterations}\n")
        
        self.classifier = MNISTClassifier(model_name=self.model_name, cache=self.cache)
        mnist_data = MNISTData()
        with tqdm(desc="Loading training data") as pbar:
            self.train_data = mnist_data.get_training_data()
            pbar.update(1)
        with tqdm(desc="Loading test data") as pbar:
            self.test_data = mnist_data.get_test_data()
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
        # print("example.digit:", example.digit, flush=True)
        # print("pred.digit:", pred.digit, flush=True)
        return str(example.digit) == str(pred.digit)

    def evaluate_base_model(self):
        """Evaluate the base model on the test data before any optimization.

        Returns:
            float: The accuracy of the base model on the test data.
        """
        print("Evaluating base model...")
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            with tqdm(total=len(self.test_data), desc="Evaluating base model") as pbar:
                futures = [executor.submit(self._evaluate_base_example, example) for example in self.test_data]
                results = []
                for future in futures:
                    results.append(future.result())
                    pbar.update(1)
                
        correct = sum(results)
        accuracy = correct / len(self.test_data)
        print(f"Base model accuracy: {accuracy:.2%}")
        return accuracy

    def _evaluate_base_example(self, example):
        """Evaluate a single example using the base classifier.

        Args:
            example: The example to evaluate.

        Returns:
            bool: True if the prediction is correct, False otherwise.
        """
        # inputs = example.inputs()
        # pred = self.classifier(inputs["pixel_matrix"])
        pred = self.classifier(example.pixel_matrix)
        # return str(example.labels()["digit"]) == str(pred.digit)
        return str(example.digit) == str(pred.digit)

    def train(self, data):
        """Train the model using the specified data.

        Args:
            data: The training data to use.

        Returns:
            The optimized classifier.
        """
        baseline_accuracy = self.evaluate_base_model()
        print(f"\nBaseline accuracy: {baseline_accuracy:.2%}")
        self.baseline_accuracy = baseline_accuracy
        
        optimizer_class = MIPROv2 if self.optimizer == 'MIPROv2' else dspy.teleprompt.BootstrapFewShot
        if self.optimizer == 'MIPROv2':
            teleprompter = optimizer_class(
                metric=self._accuracy_metric,
                num_threads=self.num_workers,
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
            
        print("Evaluating optimized model on test data...")
        print(f"Using {len(self.test_data)} test samples with {self.num_workers} threads")
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            with tqdm(total=len(self.test_data), desc="Evaluating") as pbar:
                futures = [executor.submit(self._evaluate_example, example) for example in self.test_data]
                results = []
                for future in futures:
                    results.append(future.result())
                    pbar.update(1)
                
        correct = sum(results)
        accuracy = correct / len(self.test_data)
        
        print(f"\nOptimizer: {self.optimizer}")
        print(f"Model: {self.model_name}")
        print(f"Iterations: {self.iterations}")
        print(f"Final accuracy: {accuracy:.2%}")
        
        return accuracy

    def _evaluate_example(self, example):
        """Evaluate a single example using the optimized classifier.

        Args:
            example: The example to evaluate.

        Returns:
            bool: True if the prediction is correct, False otherwise.
        """
        # inputs = example.inputs()
        # pred = self.optimized_classifier(inputs["pixel_matrix"])
        pred = self.optimized_classifier(example.pixel_matrix)
        # return str(example.labels()["digit"]) == str(pred.digit)
        return str(example.digit) == str(pred.digit)

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Train MNIST classifier')
    parser.add_argument('--optimizer', choices=['MIPROv2', 'BootstrapFewShot'], 
                      default='MIPROv2', help='Optimizer to use')
    parser.add_argument('--iterations', type=int, default=1,
                      help='Number of optimization iterations')
    parser.add_argument('--model', default="deepseek/deepseek-chat", help='Model type to use')
    parser.add_argument('--auto', choices=['light', 'medium', 'heavy'],
                      default='light', help='Auto optimization setting for MIPROv2')
    parser.add_argument('--num-workers', type=int, default=100,
                      help='Number of worker threads to use')
    parser.add_argument('--no-cache', action='store_false', dest='cache',
                      help='Disable LLM response caching')
    args = parser.parse_args()
    
    print(f"Running MNIST Trainer with {args.optimizer}")
    trainer = MNISTTrainer(
        optimizer=args.optimizer,
        iterations=args.iterations,
        model_name=args.model,
        auto=args.auto,
        num_workers=args.num_workers,
        cache=args.cache
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

    print(f"\n\nBaseline accuracy: {trainer.baseline_accuracy:.2%}")
    print(f"Final test accuracy: {accuracy:.2%}")
