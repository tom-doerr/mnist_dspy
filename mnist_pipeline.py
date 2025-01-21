from tqdm import tqdm
from dspy.teleprompt import LabeledFewShot
import dspy

class MNISTPipeline:
    """Manages training pipeline and iteration tracking"""
    def __init__(self, iterations: int, model_name: str):
        self.iterations = iterations
        self.model_name = model_name
        dspy.configure(lm=dspy.LM(model_name, cache=True))

    def run_training(self, ensemble):
        """Execute training iterations"""
        print(f"Starting ensemble boosting with {self.iterations} iterations\n")
        
        for i in range(self.iterations):
            acc = self.train_iteration(ensemble, i)
            remaining = len(ensemble.hard_examples)
            never_correct_pct = (sum(1 for ex in ensemble.hard_examples 
                                   if all(ex in hist for hist in ensemble.misclassification_history.values())) 
                                / remaining * 100) if remaining > 0 else 0
            print(f"Iteration {i+1}: Accuracy {acc:.2%} | Hard Examples: {remaining} ({remaining/100:.1%}) | Never-Correct: {never_correct_pct:.1f}%")

    def train_iteration(self, ensemble, iteration: int) -> float:
        """Train a single iteration classifier"""
        fewshot_examples = ensemble._get_hard_examples(3)
        random.shuffle(fewshot_examples)
        
        classifier = MNISTClassifier(model_name=self.model_name)
        optimizer = LabeledFewShot(k=len(fewshot_examples))
        optimized_classifier = optimizer.compile(classifier, trainset=fewshot_examples)
        ensemble.classifiers.append(optimized_classifier)
        
        evaluator = MNISTEvaluator(model_name=self.model_name, num_threads=100)
        evaluator.inference.classifier.predict = optimized_classifier.predict
        accuracy = evaluator.evaluate_accuracy(ensemble.test_pool[:100]) / 100
        
        current_hard = [ex for ex in ensemble.test_pool[:100] 
                       if ex.digit != evaluator.inference.predict(ex.pixel_matrix)]
        
        if ensemble.hard_examples:
            never_correct = [ex for ex in current_hard 
                           if all(ex in hist for hist in ensemble.misclassification_history.values())]
            persistent_hard = [ex for ex in ensemble.hard_examples if ex in current_hard]
            new_hard = list(set(ensemble.hard_examples + current_hard))
            ensemble.hard_examples = never_correct + persistent_hard + new_hard[:20]
        else:
            ensemble.hard_examples = current_hard
            
        ensemble.misclassification_history[iteration] = new_hard
        return accuracy

    def report_results(self, final_acc, results):
        """Generate final performance report"""
        print("\nRunning final ensemble evaluation...")
        print(f"\nFinal Ensemble Accuracy: {final_acc:.2%}")
        
        initial_errors = len(self.misclassification_history.get(0, []))
        final_errors = sum(1 for v in results.values() if v['majority'] != v['true_label'])
        print(f"Error Reduction: {initial_errors} → {final_errors} ({(initial_errors-final_errors)/initial_errors:.1%})")
