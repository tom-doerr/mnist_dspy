<h1 align="center"> 🔢 MNIST Classification with DSPy </h1>

<p align="center">
    <a href="https://github.com/tom-doerr/mnist_dspy/stargazers"
        ><img
            src="https://img.shields.io/github/stars/tom-doerr/mnist_dspy?colorA=2c2837&colorB=c9cbff&style=for-the-badge&logo=starship style=flat-square"
            alt="Repository's starts"
    /></a>
    <a href="https://github.com/tom-doerr/mnist_dspy/issues"
        ><img
            src="https://img.shields.io/github/issues-raw/tom-doerr/mnist_dspy?colorA=2c2837&colorB=f2cdcd&style=for-the-badge&logo=starship style=flat-square"
            alt="Issues"
    /></a>
    <a href="https://github.com/tom-doerr/mnist_dspy/blob/main/LICENSE"
        ><img
            src="https://img.shields.io/github/license/tom-doerr/mnist_dspy?colorA=2c2837&colorB=b5e8e0&style=for-the-badge&logo=starship style=flat-square"
            alt="License"
    /><br />
    <a href="https://github.com/tom-doerr/mnist_dspy/commits/main"
        ><img
            src="https://img.shields.io/github/last-commit/tom-doerr/mnist_dspy/main?colorA=2c2837&colorB=ddb6f2&style=for-the-badge&logo=starship style=flat-square"
            alt="Latest commit"
    /></a>
    <a href="https://github.com/tom-doerr/mnist_dspy"
        ><img
            src="https://img.shields.io/github/repo-size/tom-doerr/mnist_dspy?colorA=2c2837&colorB=89DCEB&style=for-the-badge&logo=starship style=flat-square"
            alt="GitHub repository size"
    /></a>
</p>

MNIST digit classification using DSPy framework with support for various LLM backends. Configure optimizers and training iterations for optimal performance.

<div align="center">
  <video src="https://private-user-images.githubusercontent.com/23431444/410143213-6fed41d1-ed43-4492-8d63-0da7f2d5ca04.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Mzg3ODMwNTgsIm5iZiI6MTczODc4Mjc1OCwicGF0aCI6Ii8yMzQzMTQ0NC80MTAxNDMyMTMtNmZlZDQxZDEtZWQ0My00NDkyLThkNjMtMGRhN2YyZDVjYTA0Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTAyMDUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwMjA1VDE5MTIzOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWM5NTdmZThhMmNkMDMxNGQzZjNkOGFlNmViYjgyODExZjI0NmE4MWYyMjA1MmU1M2E5YTdkYjQ3YTU0NGVmMzQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.ZRKhWDLOGBGzY7NarOoe0X_uIuwJhvgzZHWkhSAb1rQ" width="600" controls autoplay muted loop></video>
</div>

## 🚀 Features

- **DSPy-powered MNIST classification** with support for multiple LLM backends
- **Flexible optimizer selection** (MIPROv2, BootstrapFewShot)
- **Configurable training iterations** and worker threads
- **Model selection** (supports any LLM compatible with DSPy)
- **Response caching** for faster iterations

## 📦 Installation

```bash
git clone https://github.com/tom-doerr/mnist_dspy.git
cd mnist_dspy
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
