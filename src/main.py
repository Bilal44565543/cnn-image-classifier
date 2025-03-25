import torch
from src.data.loader import CIFAR10DataLoader
from src.models.cnn import CNNModel
from src.models.training import ModelTrainer
from src.utils.visualization import ModelVisualizer
from src.utils.metrics import ModelEvaluator
from src.config import DEVICE_CONFIG, PATHS

def main():
    # Set device
    device = torch.device(DEVICE_CONFIG['device'])
    print(f"Using device: {device}")
    
    # Initialize data loader
    print("\nInitializing data loader...")
    data_loader = CIFAR10DataLoader()
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    
    # Initialize model
    print("\nInitializing model...")
    model = CNNModel()
    model.to(device)
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = ModelTrainer(model, train_loader, val_loader)
    
    # Train the model
    print("\nStarting training...")
    trainer.train()
    
    # Initialize visualizer
    print("\nInitializing visualizer...")
    visualizer = ModelVisualizer(save_dir=PATHS['results_dir'])
    
    # Plot training history
    print("\nPlotting training history...")
    visualizer.plot_training_history(trainer.history)
    
    # Initialize evaluator
    print("\nInitializing evaluator...")
    evaluator = ModelEvaluator(save_dir=PATHS['results_dir'])
    
    # Evaluate model on test set
    print("\nEvaluating model on test set...")
    metrics, preds, labels = evaluator.evaluate_model(model, test_loader, device)
    
    # Print metrics
    evaluator.print_metrics(metrics)
    
    # Plot confusion matrix
    print("\nPlotting confusion matrix...")
    visualizer.plot_confusion_matrix(labels, preds, data_loader.class_names)
    
    # Plot feature maps for a sample image
    print("\nPlotting feature maps...")
    sample_batch = next(iter(test_loader))
    sample_image = sample_batch[0][:1].to(device)
    visualizer.plot_feature_maps(model, sample_image)
    
    # Plot class activation map
    print("\nPlotting class activation map...")
    visualizer.plot_class_activation_maps(model, sample_image, sample_batch[1][0].item())
    
    print("\nTraining and evaluation completed!")

if __name__ == "__main__":
    main() 