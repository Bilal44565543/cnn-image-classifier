# CNN Image Classifier

A deep learning project that implements a Convolutional Neural Network (CNN) for image classification using PyTorch and FastAPI. The project includes model training, evaluation, visualization, and a web interface for real-time predictions.

## Features

- **Model Architecture**: Customizable CNN implementation with configurable layers
- **Data Pipeline**: Efficient data loading and preprocessing with augmentation
- **Training**: Configurable training loop with early stopping and checkpointing
- **Evaluation**: Comprehensive metrics and visualizations
- **Web Interface**: FastAPI-based web application for real-time predictions
- **Documentation**: Detailed documentation and interactive notebooks

## Documentation

- [Technical Documentation](docs/TECHNICAL.md) - Detailed technical specifications and implementation details
- [API Documentation](docs/API.md) - API endpoints and usage guide
- [Development Guide](docs/DEVELOPMENT.md) - Guide for developers and contributors

## Project Structure

```
cnn-image-classifier/
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Model architecture and training
│   ├── utils/          # Utility functions
│   └── config.py       # Configuration settings
├── app/                # FastAPI web application
├── docs/              # Documentation
├── notebooks/         # Jupyter notebooks
├── data/             # Dataset storage
├── models/           # Saved models
├── results/          # Training results and visualizations
└── logs/             # Training logs
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/gitarber/cnn-image-classifier.git
cd cnn-image-classifier
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python -m src.main
```

This will:
- Load and preprocess the CIFAR-10 dataset
- Train the CNN model
- Generate visualizations and metrics
- Save the best model

### Running the Web Interface

```bash
python -m app.main
```

Access the web interface at `http://localhost:8000/docs`

## Configuration

The project is highly configurable through `src/config.py`:
- Dataset parameters
- Model architecture
- Training settings
- Data augmentation
- Visualization options

## Results

The training process generates:
- Model checkpoints
- Training history plots
- Confusion matrix
- Feature maps
- Class activation maps

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. See our [Development Guide](docs/DEVELOPMENT.md) for more details.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

- **Arber Shquti** - [GitHub](https://github.com/gitarber) 