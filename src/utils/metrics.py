import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
from src.config import VISUALIZATION_CONFIG

class ModelEvaluator:
    def __init__(self, save_dir='results', config=VISUALIZATION_CONFIG):
        """
        Initialize the model evaluator
        
        Args:
            save_dir (str): Directory to save metric plots
            config (dict): Visualization configuration dictionary
        """
        self.save_dir = save_dir
        self.config = config
        os.makedirs(save_dir, exist_ok=True)
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate various classification metrics
        
        Args:
            y_true (list): True labels
            y_pred (list): Predicted labels
            y_pred_proba (list, optional): Predicted probabilities
            
        Returns:
            dict: Dictionary containing various metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_pred_proba is not None:
            # Calculate ROC curve and AUC for each class
            n_classes = y_pred_proba.shape[1]
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            metrics['roc_auc'] = roc_auc
            
            if self.config['plot_training_history']:
                self.plot_roc_curves(fpr, tpr, roc_auc)
        
        return metrics
    
    def evaluate_model(self, model, data_loader, device):
        """
        Evaluate model on a data loader
        
        Args:
            model: PyTorch model
            data_loader: DataLoader for evaluation
            device: Device to run evaluation on
            
        Returns:
            tuple: (metrics, predictions, true_labels)
        """
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        metrics = self.calculate_metrics(
            all_labels,
            all_preds,
            np.array(all_probs)
        )
        
        return metrics, all_preds, all_labels
    
    def plot_roc_curves(self, fpr, tpr, roc_auc):
        """
        Plot ROC curves for each class
        
        Args:
            fpr (dict): Dictionary of false positive rates
            tpr (dict): Dictionary of true positive rates
            roc_auc (dict): Dictionary of AUC scores
        """
        plt.figure(figsize=(10, 8))
        
        for i in fpr.keys():
            plt.plot(
                fpr[i],
                tpr[i],
                label=f'Class {i} (AUC = {roc_auc[i]:.2f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Each Class')
        plt.legend(loc="lower right")
        
        if self.config['save_plots']:
            plt.savefig(os.path.join(self.save_dir, 'roc_curves.png'))
        plt.close()
    
    def print_metrics(self, metrics):
        """
        Print evaluation metrics
        
        Args:
            metrics (dict): Dictionary of metrics
        """
        print("\nModel Evaluation Metrics:")
        print("-" * 20)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        if 'roc_auc' in metrics:
            print("\nROC AUC Scores:")
            for class_idx, auc_score in metrics['roc_auc'].items():
                print(f"Class {class_idx}: {auc_score:.4f}")

if __name__ == "__main__":
    # Example usage
    from src.models.cnn import CNNModel
    from src.data.loader import CIFAR10DataLoader
    
    # Initialize data loader
    data_loader = CIFAR10DataLoader()
    _, _, test_loader = data_loader.get_data_loaders()
    
    # Initialize model
    model = CNNModel()
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate model
    metrics, preds, labels = evaluator.evaluate_model(
        model,
        test_loader,
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    # Print metrics
    evaluator.print_metrics(metrics)
