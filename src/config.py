"""
Configuration file for the CNN image classification project
"""

# Dataset Configuration
DATASET_CONFIG = {
    'name': 'CIFAR10',
    'num_classes': 10,
    'image_size': (32, 32),
    'channels': 3,
    'train_split': 0.7,
    'val_split': 0.15,
    'batch_size': 32,
    'num_workers': 2
}

# Model Configuration
MODEL_CONFIG = {
    'architecture': 'CNN',
    'conv_layers': [
        {'in_channels': 3, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1}
    ],
    'fc_layers': [
        {'in_features': 128 * 4 * 4, 'out_features': 512},
        {'in_features': 512, 'out_features': 10}
    ],
    'dropout_rate': 0.5
}

# Training Configuration
TRAINING_CONFIG = {
    'num_epochs': 50,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss_function': 'cross_entropy',
    'early_stopping_patience': 5,
    'model_save_path': 'models/saved_models',
    'best_model_name': 'best_model.pth',
    'checkpoint_frequency': 5
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    'enabled': True,
    'random_horizontal_flip': True,
    'random_rotation': 10,
    'color_jitter': {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2
    },
    'random_affine': {
        'degrees': 0,
        'translate': (0.1, 0.1)
    }
}

# Paths Configuration
PATHS = {
    'data_dir': 'data/raw',
    'processed_data_dir': 'data/processed',
    'augmented_data_dir': 'data/augmented',
    'model_dir': 'models/saved_models',
    'log_dir': 'logs',
    'results_dir': 'results'
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    'plot_training_history': True,
    'plot_confusion_matrix': True,
    'plot_class_activation_maps': True,
    'save_plots': True,
    'plot_frequency': 1  # Plot every N epochs
}

# Device Configuration
DEVICE_CONFIG = {
    'use_cuda': False,  # Set to False if no GPU is available
    'device': 'cpu'
}
