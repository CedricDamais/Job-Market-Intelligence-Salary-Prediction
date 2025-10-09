# Generation Models

This directory contains the core text generation models and utilities for the job description generation project.

## 📁 Directory Structure

- **`lstm.py`** - LSTM-based language model implementation
- **`neural_networks.py`** - Hybrid feedforward neural network models
- **`ngram.py`** - N-gram language model implementation
- **`generation_transformer_models.py`** - Transformer-based generation models
- **`recurent_neural_network.py`** - RNN implementations

## 📊 Notebooks

- **`LSTM.ipynb`** - LSTM model training and evaluation
- **`neural_network.ipynb`** - Neural network experiments
- **`ngram.ipynb`** - N-gram model development
- **`recurent_neural_network.ipynb`** - RNN model experiments
- **`dataset_analysis.ipynb`** - Data analysis and exploration

## 🎯 Model Cache

The `model_cache/` directory contains trained model checkpoints:
- `*.pt` files - PyTorch model states
- `*.pth` files - PyTorch model checkpoints

⚠️ **Note**: Model files are large and excluded from git. To use pre-trained models:
1. Train models using the provided notebooks
2. Or download pre-trained models if available

## 🎬 Visualizations

The following GIF files show model training dynamics and results:
- `learning_rate_schedulers.gif` - Learning rate scheduling visualization
- `lstm_title_conditioning.gif` - LSTM title conditioning process
- `title_conditioning*.gif` - Various title conditioning experiments
- `model_flow.gif` - Model architecture flow

## 📊 Data

The `data/` subdirectory contains:
- **`processed/`** - Preprocessed datasets (*.parquet files)
- **`optimizer_momentum.gif`** - Optimizer momentum visualization

⚠️ **Storage Note**: Large data files and visualizations are excluded from version control to keep the repository size manageable.

## 🚀 Usage

### Training a Model

```python
from models.src.generation.lstm import LSTMLanguageModelS

# Initialize and train model
model = LSTMLanguageModelS(...)
# See notebooks for complete examples
```

### Generating Text

```python
from models.src.generation.lstm import generate_description

description = generate_description(model, "Software Engineer")
```

## 📋 Requirements

- PyTorch
- transformers
- sentence-transformers
- NLTK
- Other dependencies as specified in pyproject.toml