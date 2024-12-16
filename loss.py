import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomActiveLearningLoss:
    """
    Custom loss function for active learning that combines supervised loss and pseudo-label loss.
    The loss is weighted based on prediction uncertainty for labeled samples and confidence for pseudo-labeled samples.
    
    Attributes:
        pseudo_weight (float): Weight for the pseudo-label loss.
        temperature (float): Temperature parameter for softening predictions.
        uncertainty_weight (float): Weight for uncertainty-based loss adjustment.
        criterion (torch.nn.CrossEntropyLoss): Cross entropy loss used for computing both supervised and pseudo-label losses.
    """
    
    def __init__(self, pseudo_weight=0.5, temperature=1.0, uncertainty_weight=1.0):
        """
        Initializes the CustomActiveLearningLoss object.
        
        Args:
            pseudo_weight (float): Weight for the pseudo-label loss (default is 0.5).
            temperature (float): Temperature parameter for scaling softmax output (default is 1.0).
            uncertainty_weight (float): Weight to adjust loss based on uncertainty (default is 1.0).
        """
        self.pseudo_weight = pseudo_weight
        self.temperature = temperature
        self.uncertainty_weight = uncertainty_weight
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
    def calculate_uncertainty(self, outputs):
        """
        Calculates uncertainty scores based on prediction entropy.
        
        Entropy measures the uncertainty of a prediction, with higher values indicating greater uncertainty.
        
        Args:
            outputs (torch.Tensor): Model outputs (logits) for the input data.
        
        Returns:
            torch.Tensor: Uncertainty values computed as entropy for each sample in the batch.
        """
        probs = F.softmax(outputs / self.temperature, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy
    
    def get_confidence_weights(self, outputs):
        """
        Calculates confidence-based weights for pseudo-labels based on prediction confidence.
        
        Confidence is computed as the highest probability predicted for each sample.
        
        Args:
            outputs (torch.Tensor): Model outputs (logits) for the input data.
        
        Returns:
            torch.Tensor: Confidence weights for each sample based on the predicted probabilities.
        """
        probs = F.softmax(outputs / self.temperature, dim=-1)
        confidence, _ = torch.max(probs, dim=-1)
        return confidence
    
    def __call__(self, labeled_outputs, labels, pseudo_outputs=None, pseudo_labels=None):
        """
        Computes the total loss by combining the supervised and pseudo-label losses, adjusted by uncertainty and confidence weights.
        
        Args:
            labeled_outputs (torch.Tensor): Model outputs (logits) for labeled data.
            labels (torch.Tensor): Ground truth labels for labeled data.
            pseudo_outputs (torch.Tensor, optional): Model outputs (logits) for pseudo-labeled data.
            pseudo_labels (torch.Tensor, optional): Pseudo-labels for the pseudo-labeled data.
        
        Returns:
            torch.Tensor: The total loss for the active learning task, combining supervised and pseudo-label losses.
        """
        loss = 0
        
        # Supervised loss with uncertainty weighting
        if labeled_outputs is not None and labels is not None:
            supervised_loss = self.criterion(labeled_outputs / self.temperature, labels)
            
            # Add uncertainty weighting for labeled samples
            uncertainty = self.calculate_uncertainty(labeled_outputs)
            normalized_uncertainty = uncertainty / (uncertainty.max() + 1e-10)
            weighted_supervised_loss = supervised_loss * (1 + self.uncertainty_weight * normalized_uncertainty)
            loss += weighted_supervised_loss.mean()
        
        # Pseudo-label loss with confidence weighting
        if pseudo_outputs is not None and pseudo_labels is not None:
            pseudo_loss = self.criterion(pseudo_outputs / self.temperature, pseudo_labels)
            
            # Weight pseudo-labels by prediction confidence
            confidence_weights = self.get_confidence_weights(pseudo_outputs)
            weighted_pseudo_loss = pseudo_loss * confidence_weights
            
            # Add uncertainty weighting for pseudo-labeled samples
            uncertainty = self.calculate_uncertainty(pseudo_outputs)
            normalized_uncertainty = uncertainty / (uncertainty.max() + 1e-10)
            weighted_pseudo_loss = weighted_pseudo_loss * (1 + self.uncertainty_weight * normalized_uncertainty)
            
            loss += self.pseudo_weight * weighted_pseudo_loss.mean()
        
        return loss
