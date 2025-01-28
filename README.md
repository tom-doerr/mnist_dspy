# MNIST Classification with DSPy & DeepSeek

<div align="center">

![License](https://img.shields.io/badge/license-MIT-brightgreen?style=flat-square)
![Python](https://img.shields.io/badge/python-3.8+-blue?style=flat-square&logo=python)
![DSPy](https://img.shields.io/badge/DSPy-latest-orange?style=flat-square)
![DeepSeek](https://img.shields.io/badge/DeepSeek-chat%20%7C%20reasoner-purple?style=flat-square)
![Status](https://img.shields.io/badge/status-active-success?style=flat-square)

</div>

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
