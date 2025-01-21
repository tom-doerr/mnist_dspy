# MNIST Classification with DSPy & DeepSeek

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](https://github.com/yourusername/mnist-dspy/issues)
[![Open Issues](https://img.shields.io/github/issues/yourusername/mnist-dspy.svg)](https://github.com/yourusername/mnist-dspy/issues)

Advanced MNIST classification system implementing cutting-edge techniques with DSPy and DeepSeek models. Features ensemble learning, boosting, and automated hyperparameter optimization.

## 🚀 Features

- **DSPy-powered MNIST classification** with DeepSeek models
- **Adaptive boosting** with hard example mining
- **Ensemble learning** with majority voting
- **Automated hyperparameter tuning** using random search
- **Advanced evaluation** with multi-threaded testing
- **Flexible training pipelines** supporting various optimizers (MIPROv2, BootstrapFewShot)
- **Comprehensive metrics tracking** and error analysis

## 📦 Installation

```bash
git clone https://github.com/yourusername/mnist-dspy.git
cd mnist-dspy
pip install -r requirements.txt
```

## 🧠 Key Components

```
mnist_dspy/
├── core/
│   ├── mnist_booster.py       # Adaptive boosting implementation
│   ├── mnist_ensemble.py      # Ensemble learning with majority voting
│   └── mnist_evaluation.py    # Multi-threaded evaluation
├── optimization/
│   ├── mnist_hyperparam_search.py  # Hyperparameter tuning
│   └── mnist_random_search.py      # Random search implementation
├── pipelines/
│   ├── mnist_mipro_auto.py    # Automated MIPROv2 pipeline
│   └── mnist_pipeline.py      # Main training pipeline
└── data/
    └── mnist_data.py          # Data loading and preprocessing
```

## 🏁 Basic Usage

### Train a boosted ensemble:
```python
from mnist_booster import MNISTBoosterV2

booster = MNISTBoosterV2(iterations=5)
accuracy = booster.run()
print(f"Final Accuracy: {accuracy:.2%}")
```

### Run hyperparameter search:
```bash
python mnist_random_search.py --trials 20
```

### Evaluate a model:
```python
from mnist_evaluation import MNISTEvaluator

evaluator = MNISTEvaluator()
accuracy = evaluator.run_evaluation()
print(f"Model Accuracy: {accuracy:.2%}")
```

## 📊 Results

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
