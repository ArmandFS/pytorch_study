# PyTorch Study Repository!

This repository is designed to help me learn PyTorch from the ground up, with hands-on examples and progressive tutorials. Mostly from freecodecamp!

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Installation

1. **Clone or download this repository**
   ```bash
   git clone <your-repo-url>
   cd pytorch_study
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   # Using venv
   python -m venv pytorch_env
   
   # Activate the environment
   # On Windows:
   pytorch_env\Scripts\activate
   # On macOS/Linux:
   source pytorch_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   ```

## 📁 Project Structure

```
pytorch_study/
├── notebooks/           # Jupyter notebooks for learning
│   ├── 01_pytorch_basics.ipynb
│   ├── 02_neural_networks.ipynb
│   ├── 03_computer_vision.ipynb
│   └── 04_natural_language_processing.ipynb
├── data/               # Datasets and data files
├── models/             # Saved model checkpoints
├── utils/              # Utility functions and helpers
├── examples/           # Standalone example scripts
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

## 📚 Learning Path

### 1. **PyTorch Basics** (`notebooks/01_pytorch_basics.ipynb`)
- Tensors and operations
- Automatic differentiation
- Basic neural network construction
- Training loops

### 2. **Neural Networks** (`notebooks/02_neural_networks.ipynb`)
- Building different types of neural networks
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- Transfer learning

### 3. **Computer Vision** (`notebooks/03_computer_vision.ipynb`)
- Image preprocessing and augmentation
- CNN architectures
- Object detection
- Image segmentation

### 4. **Natural Language Processing** (`notebooks/04_natural_language_processing.ipynb`)
- Text preprocessing
- Word embeddings
- RNNs and LSTMs for NLP
- Transformer models

## 🛠️ Getting Started with Jupyter

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Or start JupyterLab (recommended)**
   ```bash
   jupyter lab
   ```

3. **Navigate to the `notebooks/` folder and start with `01_pytorch_basics.ipynb`**

## 💡 Tips for Learning

1. **Start with the basics**: Work through the notebooks in order
2. **Experiment**: Modify the code, try different parameters
3. **Practice**: Create your own small projects
4. **Read the documentation**: PyTorch docs are excellent
5. **Join the community**: PyTorch forums and Discord are great resources

## 🔧 Common Commands

### Check PyTorch installation
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Create a simple tensor
```python
import torch
x = torch.tensor([1, 2, 3, 4])
print(x)
```

### Basic neural network
```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

model = SimpleNet()
```

## 📖 Additional Resources

- [PyTorch Official Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Deep Learning with PyTorch Book](https://pytorch.org/deep-learning-with-pytorch)
- [PyTorch Examples on GitHub](https://github.com/pytorch/examples)

## 🤝 Contributing

Feel free to:
- Add your own examples
- Improve existing notebooks
- Fix bugs or typos
- Suggest new topics

## 📝 License

This project is for educational purposes. Feel free to use and modify as needed.

---

Happy learning! 🎉 If you have any questions, don't hesitate to ask or check the PyTorch community forums.

