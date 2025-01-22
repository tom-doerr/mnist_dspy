#!/usr/bin/env python

import dspy
from typing import List
from mnist_data import MNISTData
from mnist_dspy import MNISTClassifier
from mnist_evaluation import MNISTEvaluator
import random
import argparse
import os
import datetime


# self.aggregator = dspy.Predict('self, predictions -> number')
aggregator = dspy.Predict('self, predictions -> number')

# classifiers: List[MNISTClassifier] = []




class MNISTBoosterV2:
    """Advanced boosting implementation with hard example tracking"""
    
    def __init__(self, iterations: int = 10, model_name: str = "deepseek/deepseek-chat"):
        self.hard_examples: List[dspy.Example] = []
        self.iterations = iterations
        self.model_name = model_name
        self.classifiers: List[MNISTClassifier] = []
        self.args = self.parse_args()
        
        # Create unique run directory
        self.run_dir = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(f"runs/{self.run_dir}", exist_ok=True)
        print(f"📁 Saving run artifacts to: runs/{self.run_dir}")

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--mipro", action="store_true", help="Use MIPRO_v2 instead of FewShot")
        return parser.parse_args()

    def run(self):
        """Execute full boosting pipeline"""
        print("⚡ Starting MNIST Boosting Process")
        booster_configs = []  # To track configs and their results
        
        test_data_full = MNISTData().get_test_data()
        random.shuffle(test_data_full)
        test_data = test_data_full[:100]  

        training_data_full = MNISTData().get_training_data()
        random.shuffle(training_data_full)
        if self.args.mipro:
            training_data = training_data_full[:1000] 
        else:
            training_data = training_data_full[:100] 
        
        for i in range(self.iterations):
            print(f"\n🚀 Starting Boosting Iteration {i+1}/{self.iterations}")
            random.shuffle(training_data_full)
            # if args.mipro:
                # training_data = training_data_full[:1000]
            
            # 1. Train new classifier on current hard examples
            classifier = MNISTClassifier(model_name=self.model_name)
            
            # 2. Get hard examples from previous iteration
            if self.hard_examples:
                print(f"📚 Training with {len(self.hard_examples)} hard examples")
                training_data = self.hard_examples
            else:  # First iteration uses random sample
                # training_data = MNISTData().get_training_data()[:100]
                print("⚠️  No hard examples found, using random sample instead")

            # Configure and compile the classifier
            # teleprompter = dspy.teleprompt.BootstrapFewShot(
                # max_bootstrapped_demos=0,
                # max_labeled_demos=3
            # )
            # num_few_shot = 10
            num_few_shot = 0
            # Store optimizer configuration
            optimizer_config = {
                'use_mipro': self.args.mipro,
                'num_few_shot': num_few_shot,
                'optimizer_class': 'MIPROv2' if self.args.mipro else 'LabeledFewShot',
                'max_labeled_demos': num_few_shot if self.args.mipro else None,
                'k': num_few_shot if not self.args.mipro else None,
                'model_name': self.model_name,
                # 'auto_setting': 'heavy',
                'auto_setting': 'light',
                'iterations': self.iterations,
            }
            
            if self.args.mipro:
                teleprompter = dspy.MIPROv2(
                    max_labeled_demos=optimizer_config['max_labeled_demos'],
                    metric=self.metric,
                    # auto='light',
                    auto=optimizer_config['auto_setting'],
                )
            else:
                teleprompter = dspy.teleprompt.LabeledFewShot(
                    k=optimizer_config['k']
                )
            sampled_data = random.sample(training_data, num_few_shot)
            #remove the sampled data from the test data
            training_data = [x for x in training_data if x not in sampled_data]
            if self.args.mipro:

                # compiled_classifier = teleprompter.compile(classifier, trainset=training_data_full, requires_permission_to_run=False)
                compiled_classifier = teleprompter.compile(classifier, trainset=training_data, requires_permission_to_run=False)
            else:
                compiled_classifier = teleprompter.compile(classifier, trainset=sampled_data)
            
            # 3. Add compiled classifier to ensemble
            self.classifiers.append(compiled_classifier)
            # Save classifier to run directory
            classifier_path = f"runs/{self.run_dir}/classifier_{i+1}.json"
            compiled_classifier.save(classifier_path)
            print(f"💾 Saved classifier to: {classifier_path}")
            
            self.hard_examples = self.get_hard_examples(training_data, classifier)
            
            # 4. Evaluate current ensemble
            accuracy = self.evaluate_ensemble_accuracy(test_data)
            print(f"🎯 Iteration {i+1} Ensemble Accuracy: {accuracy:.2%}")
            
            # Store config with accuracy
            optimizer_config['accuracy'] = accuracy
            booster_configs.append(optimizer_config)
            
            if len(self.hard_examples) <= num_few_shot:
                break

        # Print final configuration report
        print("\n=== Boosting Configuration Report ===")
        for i, config in enumerate(booster_configs):
            print(f"\nIteration {i+1}:")
            print(f"Optimizer: {config['optimizer_class']}")
            print(f"Few-shot examples: {config['num_few_shot']}")
            print(f"Model: {config['model_name']}")
            print(f"Accuracy: {config['accuracy']:.2%}")
            if config['use_mipro']:
                print(f"MIPRO max_labeled_demos: {config['max_labeled_demos']}")
                print(f"Auto setting: {config['auto_setting']}")
            else:
                print(f"LabeledFewShot k: {config['k']}")

        return accuracy
    # def metric(self, pixel_matrix: str, number: str, trace=None) -> int:
        # """Simple metric to calculate distance from correct number"""
        # pred = self.ensemble_predict(pixel_matrix)
        # print("pred.number:", pred.number)
        # print("number:", number)
        # metric_val =  1 if pred.number != number else 0
        # print("metric_val:", metric_val)
        # return metric_val

    def metric(self, true: str, pred: str, trace=None) -> int:
        # print("pred.number:", pred.number)
        # print("number:", number)
        # metric_val =  1 if pred.number != number else 0
        print("pred:", pred)
        print("true:", true)
        metric_val =  1 if pred.number == str(true.number) else 0
        print("metric_val:", metric_val)
        return metric_val

    def ensemble_predict(self, pixel_matrix: str) -> dspy.Prediction:
        # print("Ensemble predict")
        predictions = [str(clf(pixel_matrix).number) for clf in self.classifiers]
        # predictions = [str(clf(pixel_matrix).number) for clf in classifiers]
        print("predictions:", predictions)
        majority = max(set(predictions), key=predictions.count)
        print("majority:", majority)
        # number = aggregator(predictions)

        # print(f"Ensemble Prediction: {majority}")
        return dspy.Prediction(number=majority)
        # return dspy.Prediction(number=number)


    def evaluate_ensemble_accuracy(self, test_data: List[dspy.Example]) -> float:
        """Evaluate ensemble with majority voting"""
        # def ensemble_predict(pixel_matrix: str) -> dspy.Prediction:
            # predictions = [str(clf(pixel_matrix).number) for clf in self.classifiers]
            # majority = max(set(predictions), key=predictions.count)
            # return dspy.Prediction(number=majority)

        evaluator = MNISTEvaluator(model_name=self.model_name, num_threads=100)
        return evaluator.evaluate_accuracy(test_data, predictor=self.ensemble_predict)
        
    def get_hard_examples(self, training_data: List[dspy.Example], predictor) -> List[dspy.Example]:
        """Collect misclassified examples from test data"""
        from concurrent.futures import ThreadPoolExecutor
        
        self.hard_examples = []
        
        # def process_example(example):
            # pred = predictor(example.pixel_matrix)
            # if str(pred.number) != str(example.number):
                # return example
            # return None
        def process_example(example):
            pred = self.ensemble_predict(example.pixel_matrix)
            if str(pred.number) != str(example.number):
                return example
            return None
            
        # Process examples in parallel with 100 threads
        with ThreadPoolExecutor(max_workers=100) as executor:
            results = executor.map(process_example, training_data)
            
            # Collect non-None results (failed examples)
            self.hard_examples = [ex for ex in results if ex is not None]
                
        print(f"Found {len(self.hard_examples)} hard examples from {len(training_data)} total samples")
        return self.hard_examples

if __name__ == "__main__":
    # Run full boosting process with 3 iterations
    booster = MNISTBoosterV2(iterations=10)
    final_accuracy = booster.run()
    print(f"\n🏆 Final Boosted Ensemble Accuracy: {final_accuracy:.2%}")
