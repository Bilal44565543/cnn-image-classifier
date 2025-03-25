import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import MODEL_CONFIG

class CNNModel(nn.Module):
    def __init__(self, config=MODEL_CONFIG):
        """
        Initialize the CNN model
        
        Args:
            config (dict): Model configuration dictionary
        """
        super(CNNModel, self).__init__()
        self.config = config
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        for layer_config in config['conv_layers']:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=layer_config['in_channels'],
                        out_channels=layer_config['out_channels'],
                        kernel_size=layer_config['kernel_size'],
                        stride=layer_config['stride'],
                        padding=layer_config['padding']
                    ),
                    nn.BatchNorm2d(layer_config['out_channels']),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            )
        
        # Build fully connected layers
        self.fc_layers = nn.ModuleList()
        for i, layer_config in enumerate(config['fc_layers']):
            if i < len(config['fc_layers']) - 1:  # Not the last layer
                self.fc_layers.append(
                    nn.Sequential(
                        nn.Linear(
                            in_features=layer_config['in_features'],
                            out_features=layer_config['out_features']
                        ),
                        nn.ReLU(),
                        nn.Dropout(config['dropout_rate'])
                    )
                )
            else:  # Last layer
                self.fc_layers.append(
                    nn.Linear(
                        in_features=layer_config['in_features'],
                        out_features=layer_config['out_features']
                    )
                )
    
    def forward(self, x):
        """
        Forward pass of the model
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        
        return x
    
    def get_conv_output(self, x):
        """
        Get the output of the last convolutional layer
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output of the last convolutional layer
        """
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        return x
    
    def get_feature_maps(self, x):
        """
        Get feature maps from all convolutional layers
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            list: List of feature maps from each convolutional layer
        """
        feature_maps = []
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            feature_maps.append(x)
        return feature_maps

if __name__ == "__main__":
    # Example usage
    from src.config import DATASET_CONFIG
    
    # Create a sample input
    batch_size = 4
    sample_input = torch.randn(
        batch_size,
        DATASET_CONFIG['channels'],
        DATASET_CONFIG['image_size'][0],
        DATASET_CONFIG['image_size'][1]
    )
    
    # Initialize model
    model = CNNModel()
    
    # Forward pass
    output = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Get feature maps
    feature_maps = model.get_feature_maps(sample_input)
    print("\nFeature map shapes:")
    for i, feature_map in enumerate(feature_maps):
        print(f"Layer {i+1}: {feature_map.shape}")
