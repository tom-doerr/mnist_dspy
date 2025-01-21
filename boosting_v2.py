import dspy
from typing import List
from mnist_data import MNISTData
from mnist_dspy import MNISTClassifier

class MNISTBoosterV2:
    """Advanced boosting implementation with hard example tracking"""
    
    def __init__(self, iterations: int = 3, model_name: str = "deepseek/deepseek-chat"):
        self.hard_examples: List[dspy.Example] = []
        self.iterations = iterations
        self.model_name = model_name
        self.classifiers: List[MNISTClassifier] = []

    def run(self):
        """Execute full boosting pipeline"""
        print("⚡ Starting MNIST Boosting Process")
        
        test_data = MNISTData().get_test_data()[:100]  # Use first 100 test samples
        
        for i in range(self.iterations):
            print(f"\n🚀 Starting Boosting Iteration {i+1}/{self.iterations}")
            
            # 1. Train new classifier on current hard examples
            classifier = MNISTClassifier(model_name=self.model_name)
            
            # 2. Get hard examples from previous iteration
            if self.hard_examples:
                print(f"📚 Training with {len(self.hard_examples)} hard examples")
                training_data = self.hard_examples
            else:  # First iteration uses random sample
                training_data = MNISTData().get_training_data()[:100]
                print("⚠️  No hard examples found, using random sample instead")
            
            # 3. Train and add to ensemble
            self.classifiers.append(classifier)
            self.hard_examples = self.get_hard_examples(test_data, classifier)
            
            # 4. Evaluate current ensemble
            accuracy = self.evaluate_ensemble_accuracy(test_data)
            print(f"🎯 Iteration {i+1} Ensemble Accuracy: {accuracy:.2%}")

        return accuracy

    def evaluate_ensemble_accuracy(self, test_data: List[dspy.Example]) -> float:
        """Evaluate ensemble with majority voting"""
        def ensemble_predict(pixel_matrix: str) -> dspy.Prediction:
            predictions = [str(clf(pixel_matrix).number) for clf in self.classifiers]
            majority = max(set(predictions), key=predictions.count)
            return dspy.Prediction(number=majority)

        evaluator = MNISTEvaluator(model_name=self.model_name)
        return evaluator.evaluate_accuracy(test_data, predictor=ensemble_predict)
        
    def get_hard_examples(self, test_data: List[dspy.Example], predictor) -> List[dspy.Example]:
        """Collect misclassified examples from test data"""
        self.hard_examples = []
        
        for example in test_data:
            pred = predictor(example.pixel_matrix)
            if str(pred.number) != str(example.number):
                self.hard_examples.append(example)
                
        print(f"Found {len(self.hard_examples)} hard examples from {len(test_data)} total samples")
        return self.hard_examples

if __name__ == "__main__":
    # Run full boosting process with 3 iterations
    booster = MNISTBoosterV2(iterations=3)
    final_accuracy = booster.run()
    print(f"\n🏆 Final Boosted Ensemble Accuracy: {final_accuracy:.2%}")
