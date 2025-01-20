#!/usr/bin/env python3
import dspy
from dspy.teleprompt import MIPROv2
from mnist_dspy import MNISTClassifier, create_training_data, create_test_data
from mnist_evaluation import MNISTEvaluator

class MNISTTrainer:
    def __init__(self):
        self.classifier = MNISTClassifier()
        # Convert training data to dspy.Example format
        raw_train = create_training_data()
        self.train_data = [dspy.Example(pixel_matrix=pixels, digit=label).with_inputs('pixel_matrix') 
                          for pixels, label in raw_train]
        
        # Convert test data to dspy.Example format
        raw_test = create_test_data()
        self.test_data = [dspy.Example(pixel_matrix=pixels, digit=label).with_inputs('pixel_matrix')
                         for pixels, label in raw_test]
        self.evaluator = MNISTEvaluator()

    def _accuracy_metric(self, example, pred, trace=None):
        # Access the digit from the example and compare with prediction
        return str(example.digit) == str(pred.digit)

    def train(self):
        print("Initializing MIPROv2 optimizer...")
        teleprompter = MIPROv2(
            metric=self._accuracy_metric,
            max_bootstrapped_demos=3,
            max_labeled_demos=5,
            num_threads=100
        )
        
        print("Starting training with MIPROv2...")
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
        print("Evaluation completed")
        return accuracy

if __name__ == "__main__":
    trainer = MNISTTrainer()
    print("Training model...")
    trainer.train()
    accuracy = trainer.evaluate()
    print(f"Optimized model accuracy: {accuracy:.2%}")
