import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from src.config import TRAINING_CONFIG, DEVICE_CONFIG
from src.models.cnn import CNNModel

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, config=TRAINING_CONFIG):
        """
        Initialize the model trainer
        
        Args:
            model: PyTorch model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config (dict): Training configuration dictionary
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(DEVICE_CONFIG['device'])
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate']
        )
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Create model save directory
        os.makedirs(config['model_save_path'], exist_ok=True)
    
    def train_epoch(self):
        """
        Train the model for one epoch
        
        Returns:
            tuple: (epoch_loss, epoch_accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Training loop
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc='Training')):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """
        Validate the model
        
        Returns:
            tuple: (val_loss, val_accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validating'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        val_loss = total_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint
        
        Args:
            epoch (int): Current epoch number
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config['model_save_path'],
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if applicable
        if is_best:
            best_model_path = os.path.join(
                self.config['model_save_path'],
                self.config['best_model_name']
            )
            torch.save(checkpoint, best_model_path)
    
    def train(self):
        """
        Train the model for the specified number of epochs
        """
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            print(f'\nEpoch {epoch+1}/{self.config["num_epochs"]}')
            
            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print metrics
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save checkpoint
            if (epoch + 1) % self.config['checkpoint_frequency'] == 0:
                self.save_checkpoint(epoch)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, is_best=True)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config['early_stopping_patience']:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path (str): Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch']

if __name__ == "__main__":
    # Example usage
    from src.data.loader import CIFAR10DataLoader
    
    # Initialize data loader
    data_loader = CIFAR10DataLoader()
    train_loader, val_loader, _ = data_loader.get_data_loaders()
    
    # Initialize model
    model = CNNModel()
    
    # Initialize trainer
    trainer = ModelTrainer(model, train_loader, val_loader)
    
    # Train the model
    trainer.train()
