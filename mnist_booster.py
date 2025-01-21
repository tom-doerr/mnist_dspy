#!/usr/bin/env python3
import dspy
from mnist_ensemble import MNISTEnsemble
from mnist_trainer import MNISTTrainer

class MNISTBooster:
    """Orchestrates boosting iterations with hard example mining"""
    def __init__(self, model_name: str = "deepseek/deepseek-chat", iterations: int = 3):
        self.model_name = model_name
        self.iterations = iterations
        self.ensemble = MNISTEnsemble(model_name)
        self.trainer = MNISTTrainer(model_name=model_name)

    def run_boosting_iteration(self, iteration: int):
        """Run a single boosting iteration"""
        print(f"\n🚀 Starting Boosting Iteration {iteration + 1}")
        
        # 1. Get hard examples from ensemble
        hard_examples = self.ensemble._get_hard_examples()
        print(f"📚 Training with {len(hard_examples)} hard examples")
        
        # 2. Train new classifier on current hard examples
        new_clf = self.trainer.train(hard_examples)
        self.ensemble.classifiers.append(new_clf)
        
        # 3. Evaluate updated ensemble
        accuracy, results = self.ensemble.evaluate()
        print(f"🎯 Iteration {iteration + 1} Ensemble Accuracy: {accuracy:.2%}")

    def run(self):
        """Execute full boosting pipeline"""
        print("⚡ Starting MNIST Boosting Process")
        for i in range(self.iterations):
            self.run_boosting_iteration(i)
        
        # Final evaluation
        final_acc, _ = self.ensemble.evaluate()
        print(f"\n🏆 Final Boosted Ensemble Accuracy: {final_acc:.2%}")
        return final_acc

if __name__ == "__main__":
    booster = MNISTBooster(iterations=3)
    booster.run()
