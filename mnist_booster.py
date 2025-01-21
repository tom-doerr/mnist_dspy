#!/usr/bin/env python3
import dspy
from mnist_ensemble import MNISTEnsemble
from mnist_trainer import MNISTTrainer
from mnist_data import MNISTData

class MNISTBooster:
    """Orchestrates boosting iterations with hard example mining
    
    Implements adaptive boosting by:
    1. Training initial classifier on random sample
    2. For each boosting iteration:
       a. Evaluate current ensemble to find hard examples
       b. Train new classifier focused on those hard examples
       c. Add new classifier to ensemble
    3. Final predictions use majority vote across all classifiers
    
    The hard example mining prioritizes:
    - Examples consistently misclassified across iterations
    - Recent errors showing new failure patterns
    - Balanced sampling of different error types"""
    def __init__(self, model_name: str = "deepseek/deepseek-chat", iterations: int = 3):
        self.model_name = model_name
        self.iterations = iterations
        self.ensemble = MNISTEnsemble(model_name)
        self.trainer = MNISTTrainer()

    def run_boosting_iteration(self, iteration: int):
        """Run a single boosting iteration"""
        print(f"\n🚀 Starting Boosting Iteration {iteration + 1}")
        
        # 1. Get hard examples from ensemble
        hard_examples = self.ensemble._get_hard_examples(num_samples=100)
        if not hard_examples:  # Fallback to random sample if no hard examples
            hard_examples = MNISTData().get_training_data()[:100]  # Uses cached dataset
            print("⚠️  No hard examples found, using random sample instead")
        print(f"📚 Training with {len(hard_examples)} hard examples")
        
        # 2. Train new classifier on current hard examples
        new_clf = self.trainer.train(hard_examples)
        self.ensemble.classifiers.append(new_clf)
        
        # 3. Evaluate updated ensemble
        accuracy = self.ensemble.evaluate()
        print(f"🎯 Iteration {iteration + 1} Ensemble Accuracy: {accuracy:.2%}")

    def run(self):
        """Execute full boosting pipeline"""
        print("⚡ Starting MNIST Boosting Process")
        
        # Start with base classifier trained on random sample
        initial_data = MNISTData().get_training_data()[:100]  # Uses cached dataset
        base_clf = self.trainer.train(initial_data)
        self.ensemble.classifiers.append(base_clf)
        
        for i in range(self.iterations):
            self.run_boosting_iteration(i)
        
        # Final evaluation
        final_acc = self.ensemble.evaluate()
        print(f"\n🏆 Final Boosted Ensemble Accuracy: {final_acc:.2%}")
        return final_acc

if __name__ == "__main__":
    booster = MNISTBooster(iterations=3)
    booster.run()
