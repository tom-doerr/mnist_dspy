# MNIST Classification with DSPy & DeepSeek

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

MNIST digit classification using DSPy and DeepSeek models, with configurable optimizers and training iterations.

## 🚀 Features

- **DSPy-powered MNIST classification** with DeepSeek models
- **Flexible optimizer selection** (MIPROv2, BootstrapFewShot)
- **Configurable training iterations**
- **Model selection** (DeepSeek Chat or Reasoner)

## 📦 Installation

```bash
git clone https://github.com/yourusername/mnist-dspy.git
cd mnist-dspy
pip install -r requirements.txt
```

## 🧠 Project Structure

```
mnist_dspy/
├── mnist_trainer.py    # Main training script with optimizer selection
├── mnist_data.py       # MNIST data loading and preprocessing
└── mnist_dspy.py       # DSPy model definitions
```

## 🏁 Basic Usage

Train a model with specific optimizer and iterations:
```bash
# Use MIPROv2 optimizer with 3 iterations
python mnist_trainer.py --optimizer MIPROv2 --iterations 3 --model reasoner

# Use BootstrapFewShot with 5 iterations
python mnist_trainer.py --optimizer BootstrapFewShot --iterations 5 --model chat
```

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
