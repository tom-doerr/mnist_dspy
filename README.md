<h1 align="center"> 🔢 MNIST Classification with DSPy </h1>

<p align="center">
    <a href="https://github.com/tom-doerr/mnist-dspy/stargazers"
        ><img
            src="https://img.shields.io/github/stars/tom-doerr/mnist-dspy?colorA=2c2837&colorB=c9cbff&style=for-the-badge&logo=starship style=flat-square"
            alt="Repository's starts"
    /></a>
    <a href="https://github.com/tom-doerr/mnist-dspy/issues"
        ><img
            src="https://img.shields.io/github/issues-raw/tom-doerr/mnist-dspy?colorA=2c2837&colorB=f2cdcd&style=for-the-badge&logo=starship style=flat-square"
            alt="Issues"
    /></a>
    <a href="https://github.com/tom-doerr/mnist-dspy/blob/main/LICENSE"
        ><img
            src="https://img.shields.io/github/license/tom-doerr/mnist-dspy?colorA=2c2837&colorB=b5e8e0&style=for-the-badge&logo=starship style=flat-square"
            alt="License"
    /><br />
    <a href="https://github.com/tom-doerr/mnist-dspy/commits/main"
        ><img
            src="https://img.shields.io/github/last-commit/tom-doerr/mnist-dspy/main?colorA=2c2837&colorB=ddb6f2&style=for-the-badge&logo=starship style=flat-square"
            alt="Latest commit"
    /></a>
    <a href="https://github.com/tom-doerr/mnist-dspy"
        ><img
            src="https://img.shields.io/github/repo-size/tom-doerr/mnist-dspy?colorA=2c2837&colorB=89DCEB&style=for-the-badge&logo=starship style=flat-square"
            alt="GitHub repository size"
    /></a>
</p>

MNIST digit classification using DSPy framework with support for various LLM backends. Configure optimizers and training iterations for optimal performance.

<p align="center">
  <video width="600" controls>
    <source src="https://raw.githubusercontent.com/tom-doerr/bins/main/mnist_dspy/output.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</p>

## 🚀 Features

- **DSPy-powered MNIST classification** with support for multiple LLM backends
- **Flexible optimizer selection** (MIPROv2, BootstrapFewShot)
- **Configurable training iterations** and worker threads
- **Model selection** (supports any LLM compatible with DSPy)
- **Response caching** for faster iterations

## 📦 Installation

```bash
git clone https://github.com/tom-doerr/mnist-dspy.git
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
# Use MIPROv2 optimizer with light optimization
python mnist_trainer.py --optimizer MIPROv2 --iterations 1 --model your-llm-model --auto light

# Use MIPROv2 with medium optimization and caching disabled
python mnist_trainer.py --optimizer MIPROv2 --iterations 1 --model your-llm-model --auto medium --no-cache

# Use BootstrapFewShot optimizer with custom number of workers
python mnist_trainer.py --optimizer BootstrapFewShot --iterations 1 --model your-llm-model --num-workers 50
```

## 🔧 Configuration Options

- `--optimizer`: Choose between 'MIPROv2' or 'BootstrapFewShot' (default: MIPROv2)
- `--iterations`: Number of optimization iterations (default: 1)
- `--model`: Model identifier for your LLM
- `--auto`: Optimization intensity for MIPROv2 ['light', 'medium', 'heavy'] (default: light)
- `--num-workers`: Number of worker threads (default: 100)
- `--no-cache`: Disable LLM response caching

## 📊 Data Processing

The trainer:
- Uses a subset of MNIST data for faster experimentation
- Processes 10,000 training examples
- Evaluates on 200 test examples
- Supports multi-threaded evaluation

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
