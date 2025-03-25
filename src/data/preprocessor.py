import torch
from torchvision import transforms
import numpy as np
from PIL import Image

class DataPreprocessor:
    def __init__(self, augment=True):
        """
        Initialize the data preprocessor
        
        Args:
            augment (bool): Whether to apply data augmentation
        """
        self.augment = augment
        
        # Define basic transformations
        self.basic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Define augmentation transformations
        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def preprocess_image(self, image):
        """
        Preprocess a single image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            torch.Tensor: Preprocessed image
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if self.augment:
            return self.augment_transform(image)
        return self.basic_transform(image)
    
    def preprocess_batch(self, batch):
        """
        Preprocess a batch of images
        
        Args:
            batch: Tuple of (images, labels) or just images
            
        Returns:
            tuple: (preprocessed_images, labels) or preprocessed_images
        """
        if isinstance(batch, tuple):
            images, labels = batch
        else:
            images = batch
            labels = None
        
        # Convert to list of images if batch is a tensor
        if isinstance(images, torch.Tensor):
            images = [img for img in images]
        
        # Preprocess each image
        processed_images = [self.preprocess_image(img) for img in images]
        
        # Stack back into a batch
        processed_batch = torch.stack(processed_images)
        
        if labels is not None:
            return processed_batch, labels
        return processed_batch
    
    @staticmethod
    def denormalize_image(image):
        """
        Denormalize an image back to original scale
        
        Args:
            image: Normalized image tensor
            
        Returns:
            numpy.ndarray: Denormalized image
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Denormalize
        image = image * 0.5 + 0.5
        image = np.clip(image, 0, 1)
        
        # Convert to uint8
        image = (image * 255).astype(np.uint8)
        
        return image

if __name__ == "__main__":
    # Example usage
    from PIL import Image
    import numpy as np
    
    # Create a sample image
    sample_image = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    sample_image = Image.fromarray(sample_image)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(augment=True)
    
    # Preprocess single image
    processed_image = preprocessor.preprocess_image(sample_image)
    print(f"Processed image shape: {processed_image.shape}")
    
    # Create a batch of images
    batch_images = torch.stack([processed_image] * 4)
    batch_labels = torch.tensor([0, 1, 2, 3])
    
    # Preprocess batch
    processed_batch, processed_labels = preprocessor.preprocess_batch((batch_images, batch_labels))
    print(f"Processed batch shape: {processed_batch.shape}")
    print(f"Processed labels: {processed_labels}")
