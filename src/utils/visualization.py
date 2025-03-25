import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from src.config import VISUALIZATION_CONFIG
import os

class ModelVisualizer:
    def __init__(self, save_dir='results', config=VISUALIZATION_CONFIG):
        """
        Initialize the model visualizer
        
        Args:
            save_dir (str): Directory to save plots
            config (dict): Visualization configuration dictionary
        """
        self.save_dir = save_dir
        self.config = config
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_training_history(self, history):
        """
        Plot training and validation metrics
        
        Args:
            history (dict): Dictionary containing training history
        """
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Training Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        
        if self.config['save_plots']:
            plt.savefig(os.path.join(self.save_dir, 'training_history.png'))
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        """
        Plot confusion matrix
        
        Args:
            y_true (list): True labels
            y_pred (list): Predicted labels
            class_names (list): List of class names
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        if self.config['save_plots']:
            plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'))
        plt.close()
    
    def plot_feature_maps(self, model, image, layer_idx=None):
        """
        Plot feature maps from convolutional layers
        
        Args:
            model: PyTorch model
            image (torch.Tensor): Input image
            layer_idx (int, optional): Specific layer to visualize
        """
        model.eval()
        with torch.no_grad():
            feature_maps = model.get_feature_maps(image)
        
        if layer_idx is not None:
            feature_maps = [feature_maps[layer_idx]]
        
        for i, feature_map in enumerate(feature_maps):
            # Get the number of channels
            n_channels = feature_map.size(1)
            
            # Create a grid of feature maps
            n_cols = 8
            n_rows = (n_channels + n_cols - 1) // n_cols
            
            plt.figure(figsize=(15, 2 * n_rows))
            plt.suptitle(f'Feature Maps - Layer {i+1}')
            
            for j in range(min(n_channels, n_rows * n_cols)):
                plt.subplot(n_rows, n_cols, j + 1)
                plt.imshow(feature_map[0, j].cpu().numpy(), cmap='viridis')
                plt.axis('off')
            
            plt.tight_layout()
            
            if self.config['save_plots']:
                plt.savefig(os.path.join(self.save_dir, f'feature_maps_layer_{i+1}.png'))
            plt.close()
    
    def plot_class_activation_maps(self, model, image, target_class):
        """
        Plot class activation maps
        
        Args:
            model: PyTorch model
            image (torch.Tensor): Input image
            target_class (int): Target class index
        """
        model.eval()
        with torch.no_grad():
            # Get the last convolutional layer output
            conv_output = model.get_conv_output(image)
            
            # Get the weights of the last fully connected layer
            fc_weights = model.fc_layers[-1].weight.data
            
            # Calculate class activation map
            cam = torch.zeros(conv_output.size(2), conv_output.size(3))
            for i in range(conv_output.size(1)):
                cam += conv_output[0, i] * fc_weights[target_class, i]
            
            # Normalize CAM
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        # Plot original image and CAM
        plt.figure(figsize=(10, 4))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(image[0].cpu().permute(1, 2, 0))
        plt.title('Original Image')
        plt.axis('off')
        
        # Class activation map
        plt.subplot(1, 2, 2)
        plt.imshow(cam.cpu().numpy(), cmap='jet')
        plt.title('Class Activation Map')
        plt.axis('off')
        
        plt.tight_layout()
        
        if self.config['save_plots']:
            plt.savefig(os.path.join(self.save_dir, f'cam_class_{target_class}.png'))
        plt.close()

if __name__ == "__main__":
    # Example usage
    from src.models.cnn import CNNModel
    from src.data.loader import CIFAR10DataLoader
    
    # Initialize data loader and get a sample batch
    data_loader = CIFAR10DataLoader()
    train_loader, _, _ = data_loader.get_data_loaders()
    sample_batch = next(iter(train_loader))
    images, labels = sample_batch
    
    # Initialize model
    model = CNNModel()
    
    # Initialize visualizer
    visualizer = ModelVisualizer()
    
    # Create sample history
    history = {
        'train_loss': [0.5, 0.4, 0.3],
        'train_acc': [60, 70, 80],
        'val_loss': [0.6, 0.5, 0.4],
        'val_acc': [55, 65, 75]
    }
    
    # Plot training history
    visualizer.plot_training_history(history)
    
    # Plot feature maps
    visualizer.plot_feature_maps(model, images[:1])
    
    # Plot class activation map
    visualizer.plot_class_activation_maps(model, images[:1], labels[0].item())
