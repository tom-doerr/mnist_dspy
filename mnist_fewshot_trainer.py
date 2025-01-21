#!/usr/bin/env python3
import dspy
import argparse
from dspy.teleprompt import LabeledFewShot
from mnist_dspy import MNISTClassifier, create_training_data, create_test_data
from mnist_evaluation import MNISTEvaluator
from mnist_data import MNISTData

class MNISTFewShotTrainer:
    """Train MNIST classifier using standard few-shot learning without bootstrapping"""
    
    def __init__(self, model_name: str = "deepseek/deepseek-chat", num_examples: int = 10):
        self.model_name = model_name
        self.num_examples = num_examples
        self.classifier = MNISTClassifier(model_name)
        self.data = MNISTData()
        self.teleprompter = dspy.teleprompt.BootstrapFewShot(
            max_bootstrapped_demos=3,
            max_labeled_demos=3
        )
        
        # Load and prepare few-shot examples
        raw_train = create_training_data()
        self.train_data = [
            dspy.Example(pixel_matrix=e['pixel_matrix'], number=str(e['number'])).with_inputs('pixel_matrix')
            for e in raw_train[:num_examples]
        ]
        
        # Load test data
        raw_test = create_test_data()
        self.test_data = [dspy.Example(pixel_matrix=e['pixel_matrix'], number=str(e['number'])).with_inputs('pixel_matrix') for e in raw_test]
        
        self.evaluator = MNISTEvaluator(model_name=model_name)

    def train_with_hard_examples(self, hard_examples: List[dspy.Example]) -> MNISTClassifier:
        """Train classifier on challenging examples using few-shot learning"""
        if len(hard_examples) < 3:
            raise ValueError("Need at least 3 hard examples for few-shot training")
            
        # Sample exactly 3 hard examples
        training_sample = random.sample(hard_examples, 3)
        
        # Train with bootstrap few-shot on hard examples
        classifier = MNISTClassifier()
        return self.teleprompter.compile(classifier, trainset=training_sample)

    def train(self) -> MNISTClassifier:
        """Train using simple few-shot learning"""
        print(f"Training with {self.num_examples} few-shot examples...")
        
        # Initialize and configure few-shot optimizer
        optimizer = LabeledFewShot(k=self.num_examples)
        self.optimized_classifier = optimizer.compile(
            self.classifier,
            trainset=self.train_data
        )
        
        return self.optimized_classifier

    def evaluate(self) -> float:
        """Evaluate model on test set"""
        if not hasattr(self, 'optimized_classifier'):
            raise ValueError("Model must be trained before evaluation")
            
        print("Evaluating on test set...")
        print("self.test_data:", self.test_data)
        input('Press Enter to continue...')
        return self.evaluator.evaluate_accuracy(self.test_data, predictor=self.optimized_classifier)

def parse_args():
    parser = argparse.ArgumentParser(description='Train MNIST classifier with Few-Shot learning')
    parser.add_argument('--model', choices=['reasoner', 'chat'], default='chat',
                      help='Base model: reasoner (specialized) or chat (general purpose)')
    parser.add_argument('--examples', type=int, default=10,
                      help='Number of few-shot examples to use')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model_name = 'deepseek/deepseek-reasoner' if args.model == 'reasoner' else 'deepseek/deepseek-chat'
    
    print(f"Running MNIST Few-Shot Trainer with {args.examples} examples")
    trainer = MNISTFewShotTrainer(model_name=model_name, num_examples=args.examples)
    trainer.train()
    accuracy = trainer.evaluate()
    print(f"\nFew-Shot Model Accuracy: {accuracy:.2%}")
