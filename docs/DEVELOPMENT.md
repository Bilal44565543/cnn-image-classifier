# Development Guide

## Development Setup

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

3. Install development dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
cnn-image-classifier/
├── src/                    # Source code
│   ├── data/              # Data handling
│   ├── models/            # Model implementation
│   ├── utils/             # Utility functions
│   └── config.py          # Configuration
├── app/                   # FastAPI application
├── docs/                  # Documentation
├── notebooks/             # Jupyter notebooks
├── tests/                 # Test files
├── data/                  # Data storage
├── models/                # Saved models
├── results/               # Training results
└── logs/                  # Log files
```

## Code Style

This project follows PEP 8 guidelines. Key points:
- Use 4 spaces for indentation
- Maximum line length: 79 characters
- Use descriptive variable names
- Add docstrings to all functions and classes
- Use type hints

Example:
```python
def process_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Process an image by resizing and normalizing it.
    
    Args:
        image: Input image as numpy array
        size: Target size (width, height)
        
    Returns:
        Processed image as numpy array
    """
    # Implementation
```

## Testing

1. Run tests:
```bash
python -m pytest tests/
```

2. Run with coverage:
```bash
python -m pytest --cov=src tests/
```

## Adding New Features

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes following the code style guidelines

3. Add tests for new functionality

4. Update documentation

5. Submit a pull request

## Model Development

### Adding New Model Architectures

1. Create a new model class in `src/models/`
2. Implement the required methods:
   - `__init__`
   - `forward`
   - `predict`

Example:
```python
class NewModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        # Implementation
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implementation
        
    def predict(self, x: torch.Tensor) -> Dict:
        # Implementation
```

### Training Process

1. Configure model parameters in `src/config.py`
2. Run training:
```bash
python -m src.main
```

3. Monitor training progress in `logs/`
4. View results in `results/`

## API Development

### Adding New Endpoints

1. Create new route in `app/main.py`
2. Add request/response models
3. Implement endpoint logic
4. Update API documentation

Example:
```python
@app.post("/new-endpoint")
async def new_endpoint(request: RequestModel) -> ResponseModel:
    # Implementation
```

## Documentation

### Updating Documentation

1. Update relevant files in `docs/`
2. Keep documentation in sync with code changes
3. Add examples where appropriate

### Adding New Documentation

1. Create new markdown file in `docs/`
2. Update `README.md` with links to new documentation
3. Follow existing documentation style

## Deployment

### Local Deployment

1. Train the model:
```bash
python -m src.main
```

2. Start the API server:
```bash
python -m app.main
```

### Production Deployment

1. Set up environment variables
2. Configure logging
3. Set up monitoring
4. Deploy using Docker

## Troubleshooting

### Common Issues

1. CUDA Errors
   - Check GPU availability
   - Verify CUDA installation
   - Update device configuration

2. Memory Issues
   - Reduce batch size
   - Enable gradient checkpointing
   - Monitor memory usage

3. Training Issues
   - Check learning rate
   - Verify data preprocessing
   - Monitor loss values

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

### Pull Request Process

1. Update documentation
2. Add tests
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Request review

## Release Process

1. Update version in `setup.py`
2. Update CHANGELOG.md
3. Create release tag
4. Deploy to production

## Support

For support:
1. Check documentation
2. Open an issue
3. Contact maintainers 