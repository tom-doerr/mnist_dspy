from dspy.teleprompt import MIPROv2
from mnist_dspy import MNISTClassifier, create_training_data, create_test_data
from mnist_evaluation import MNISTEvaluator

class MNISTTrainer:
    def __init__(self):
        self.classifier = MNISTClassifier()
        self.train_data = create_training_data()
        self.test_data = create_test_data()
        self.evaluator = MNISTEvaluator()

    def _accuracy_metric(self, example, pred, trace=None):
        return example.digit == pred.digit

    def train(self):
        teleprompter = MIPROv2(
            metric=self._accuracy_metric,
            max_bootstrapped_demos=3,
            max_labeled_demos=5,
            requires_permission_to_run=False
        )
        
        self.optimized_classifier = teleprompter.compile(
            self.classifier,
            trainset=self.train_data
        )
        
        return self.optimized_classifier

    def evaluate(self):
        if not hasattr(self, 'optimized_classifier'):
            raise ValueError("Model must be trained before evaluation")
            
        return self.evaluator.evaluate_accuracy(self.test_data)

if __name__ == "__main__":
    trainer = MNISTTrainer()
    print("Training model...")
    trainer.train()
    accuracy = trainer.evaluate()
    print(f"Optimized model accuracy: {accuracy:.2%}")
