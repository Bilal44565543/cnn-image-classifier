import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

class CIFAR10DataLoader:
    def __init__(self, data_dir='data/raw', batch_size=32, train_split=0.7, val_split=0.15):
        """
        Initialize the CIFAR-10 data loader
        
        Args:
            data_dir (str): Directory to store/load the dataset
            batch_size (int): Batch size for training
            train_split (float): Proportion of data to use for training
            val_split (float): Proportion of data to use for validation
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Define data transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load the dataset
        self.dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=self.transform
        )
        
        # Calculate split sizes
        total_size = len(self.dataset)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size
        
        # Split the dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        # Class names for CIFAR-10
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    
    def get_data_loaders(self):
        """
        Return all data loaders
        
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_dataset_sizes(self):
        """
        Return the sizes of all datasets
        
        Returns:
            tuple: (train_size, val_size, test_size)
        """
        return len(self.train_dataset), len(self.val_dataset), len(self.test_dataset)

if __name__ == "__main__":
    # Example usage
    data_loader = CIFAR10DataLoader()
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    train_size, val_size, test_size = data_loader.get_dataset_sizes()
    
    print(f"Dataset sizes:")
    print(f"Training: {train_size}")
    print(f"Validation: {val_size}")
    print(f"Testing: {test_size}")
    
    # Print a sample batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
        break
