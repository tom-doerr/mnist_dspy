#!/usr/bin/env python3
import argparse
from tabulate import tabulate
from mnist_mipro_auto import MNISTMIPROAutoTrainer

class MNISTHyperparamSearch:
    def __init__(self):
        self.results = []
        
    def run(self):
        # Test all combinations of models and optimization levels
        models = ['reasoner', 'chat']
        auto_levels = ['light', 'medium', 'heavy']
        
        for model in models:
            for auto_level in auto_levels:
                print(f"\n=== Starting run: model={model}, auto={auto_level} ===")
                
                trainer = MNISTMIPROAutoTrainer(
                    auto_setting=auto_level,
                    model_name='deepseek/deepseek-reasoner' if model == 'reasoner' else 'deepseek/deepseek-chat'
                )
                trainer.train()
                accuracy = trainer.evaluate()
                
                self.results.append({
                    'model': model,
                    'auto_level': auto_level,
                    'baseline_accuracy': trainer.run_config['baseline_accuracy'],
                    'final_accuracy': trainer.run_config['final_accuracy'],
                    'config': trainer.run_config
                })
        
        self._print_results()
        return self.results
    
    def _print_results(self):
        # Sort results by final accuracy descending
        sorted_results = sorted(self.results, key=lambda x: x['final_accuracy'], reverse=True)
        
        # Prepare table data
        table_data = [(
            f"{res['model']}-{res['auto_level']}",
            f"{res['baseline_accuracy']:.2%}",
            f"{res['final_accuracy']:.2%}",
            res['config']['num_threads'],
            res['config']['train_samples']
        ) for res in sorted_results]
        
        # Print formatted results
        print("\n=== Hyperparameter Search Results ===")
        print(tabulate(table_data, 
                     headers=['Configuration', 'Baseline', 'Final', 'Threads', 'Train Samples'],
                     tablefmt='rounded_outline',
                     floatfmt=".2%"))
        
def parse_args():
    parser = argparse.ArgumentParser(description='Run hyperparameter search for MNIST classification')
    parser.add_argument('--verbose', action='store_true', help='Show detailed config for each run')
    return parser.parse_args()

def main():
    args = parse_args()
    searcher = MNISTHyperparamSearch()
    results = searcher.run()
    
    if args.verbose:
        print("\n=== Detailed Configurations ===")
        for res in results:
            print(f"\n{res['model']}-{res['auto_level']}:")
            for k, v in res['config'].items():
                print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
