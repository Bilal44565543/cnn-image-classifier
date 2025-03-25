# Technical Documentation

## Model Architecture

The CNN model is implemented with the following architecture:

### Convolutional Layers
1. First Conv Layer:
   - Input: 3 channels (RGB)
   - Output: 32 channels
   - Kernel: 3x3
   - Padding: 1
   - Activation: ReLU
   - Batch Normalization

2. Second Conv Layer:
   - Input: 32 channels
   - Output: 64 channels
   - Kernel: 3x3
   - Padding: 1
   - Activation: ReLU
   - Batch Normalization

3. Third Conv Layer:
   - Input: 64 channels
   - Output: 128 channels
   - Kernel: 3x3
   - Padding: 1
   - Activation: ReLU
   - Batch Normalization

### Fully Connected Layers
1. First FC Layer:
   - Input: 128 * 4 * 4
   - Output: 512
   - Activation: ReLU
   - Dropout: 0.5

2. Output Layer:
   - Input: 512
   - Output: 10 (number of classes)
   - Activation: Softmax

## Data Pipeline

### Data Loading
- Dataset: CIFAR-10
- Image Size: 32x32
- Channels: 3 (RGB)
- Train/Val/Test Split: 70/15/15

### Preprocessing
1. Normalization:
   - Mean: [0.4914, 0.4822, 0.4465]
   - Std: [0.2023, 0.1994, 0.2010]

2. Data Augmentation:
   - Random Horizontal Flip
   - Random Rotation (±10 degrees)
   - Color Jittering
   - Random Affine Transformations

## Training Process

### Optimization
- Optimizer: Adam
- Learning Rate: 0.001
- Loss Function: Cross Entropy

### Training Loop
1. Forward Pass
2. Loss Calculation
3. Backward Pass
4. Parameter Update
5. Validation
6. Early Stopping

### Checkpointing
- Model saved every 5 epochs
- Best model saved based on validation accuracy
- Training history preserved

## Evaluation Metrics

1. Classification Metrics:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - ROC AUC

2. Visualizations:
   - Training History
   - Confusion Matrix
   - Feature Maps
   - Class Activation Maps

## Web Interface

### FastAPI Endpoints
1. `/`: Welcome message
2. `/predict`: Image classification endpoint
   - Input: Image file
   - Output: Class prediction with confidence

### API Response Format
```json
{
    "class": "predicted_class",
    "confidence": 0.95,
    "top_3_predictions": [
        {"class": "class1", "confidence": 0.95},
        {"class": "class2", "confidence": 0.03},
        {"class": "class3", "confidence": 0.02}
    ]
}
```

## Performance Considerations

1. GPU Acceleration:
   - CUDA support for faster training
   - Automatic device selection

2. Memory Management:
   - Batch processing
   - Efficient data loading
   - Gradient checkpointing

3. Scalability:
   - Modular architecture
   - Configurable parameters
   - Extensible design 