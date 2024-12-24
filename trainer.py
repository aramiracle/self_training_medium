import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm
from loss import CustomActiveLearningLoss
from preprocess import AugmentationTransforms

class SelfTraining:
    """
    A class implementing self-training for semi-supervised learning.

    This class implements a self-training algorithm that utilizes both labeled and unlabeled data
    to improve model performance. It includes features like pseudo-labeling, data augmentation,
    and uncertainty-based sample selection.

    Args:
        model: The neural network model to be trained
        optimizer: The optimizer to be used for training
        confidence_threshold (float): Threshold for pseudo-label selection (default: 0.9)
        batch_size (int): Size of mini-batches (default: 256)
        device (str): Device to run the model on ('cpu' or 'cuda') (default: 'cpu')
        pseudo_weight (float): Weight for pseudo-labeled loss (default: 0.3)
        temperature (float): Temperature parameter for softmax (default: 1.0)
        uncertainty_weight (float): Weight for uncertainty in loss calculation (default: 1.0)
        model_save_path (str): Path to save the best model (default: 'best_model.pth')
    """

    def __init__(self, model, optimizer, confidence_threshold=0.9, batch_size=256,
                 device="cpu", pseudo_weight=0.3, temperature=1.0, uncertainty_weight=1.0,
                 model_save_path='best_model.pth'):
        self.model = model
        self.custom_loss = CustomActiveLearningLoss(
            pseudo_weight=pseudo_weight,
            temperature=temperature,
            uncertainty_weight=uncertainty_weight
        )
        self.optimizer = optimizer
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.device = device
        self.best_model_state = None
        self.best_accuracy = 0
        self.model_save_path = model_save_path
        self.augmenter = AugmentationTransforms()
        self.pseudo_loader = None
        self.pseudo_data = None
        self.pseudo_targets = None

    def save_model(self):
        """
        Save the current model state to disk.

        Saves model state dict, optimizer state dict, and best accuracy achieved.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
        }, self.model_save_path)
        print(f"Model saved to {self.model_save_path} with accuracy: {self.best_accuracy:.2f}%")

    def load_best_model(self):
        """
        Load the best model state from disk.

        Loads model state dict, optimizer state dict, and sets best accuracy.
        """
        checkpoint = torch.load(self.model_save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_accuracy = checkpoint['best_accuracy']
        print(f"Loaded best model with accuracy: {self.best_accuracy:.2f}%")

    def augment_data(self, data, is_labeled=True):
        """
        Apply data augmentation to a batch of images.

        Args:
            data (torch.Tensor): Batch of images to augment
            is_labeled (bool): Whether the data is labeled (determines augmentation strength)

        Returns:
            torch.Tensor: Augmented batch of images
        """
        augmented_data = []
        for img in data:
            if is_labeled:
                aug_img = self.augmenter.weak_transform(img)
            else:
                aug_img = self.augmenter.strong_transform(img)
            augmented_data.append(aug_img)
        return torch.stack(augmented_data).to(self.device)

    def calculate_accuracy(self, loader):
        """
        Calculate prediction accuracy on a given data loader.

        Args:
            loader (DataLoader): DataLoader containing the evaluation data

        Returns:
            float: Accuracy percentage
        """
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for data, labels in loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    def evaluate_model(self, test_loader):
        """
        Perform detailed evaluation of the model on test data.

        Args:
            test_loader (DataLoader): DataLoader containing the test data

        Returns:
            tuple: (overall_accuracy, class_accuracy_dict)
        """
        self.model.eval()
        correct = 0
        total = 0
        class_correct = [0] * 10
        class_total = [0] * 10
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, labels in tqdm(test_loader, desc="Evaluating on test set"):
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                c = (predicted == labels).squeeze()
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        overall_accuracy = 100 * correct / total

        class_accuracy = {}
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

        print("\nTest Set Evaluation Results:")
        print(f"Overall Accuracy: {overall_accuracy:.2f}%")
        print("\nPer-class Accuracy:")
        for i in range(10):
            if class_total[i] > 0:
                accuracy = 100 * class_correct[i] / class_total[i]
                print(f'{classes[i]}: {accuracy:.2f}%')
                class_accuracy[classes[i]] = accuracy

        return overall_accuracy, class_accuracy

    def train_on_labeled(self, labeled_loader, epoch):
        """
        Train the model on labeled data and pseudo-labeled data if available.

        Args:
            labeled_loader (DataLoader): DataLoader containing labeled training data
            epoch (int): Current training epoch

        Returns:
            float: Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(labeled_loader, desc=f"[Epoch {epoch}] Training", leave=False)

        for data, labels in progress_bar:
            data, labels = data.to(self.device), labels.to(self.device)
            
            augmented_data = self.augment_data(data, is_labeled=True)
            outputs = self.model(augmented_data)
            
            self.optimizer.zero_grad()
            
            labeled_loss = self.custom_loss(outputs, labels)

            if self.pseudo_loader is not None:
                pseudo_loss = 0
                pseudo_count = 0

                for pseudo_data, pseudo_labels in self.pseudo_loader:
                    pseudo_data, pseudo_labels = pseudo_data.to(self.device), pseudo_labels.to(self.device)
                    
                    augmented_pseudo = self.augment_data(pseudo_data, is_labeled=False)
                    pseudo_outputs = self.model(augmented_pseudo)
                    
                    batch_pseudo_loss = self.custom_loss(None, None, pseudo_outputs, pseudo_labels)
                    
                    pseudo_loss += batch_pseudo_loss
                    pseudo_count += 1

                if pseudo_count > 0:
                    labeled_loss += pseudo_loss / pseudo_count

            labeled_loss.backward()
            self.optimizer.step()

            total_loss += labeled_loss.item()
            progress_bar.set_postfix({"Batch Loss": labeled_loss.item()})

        return total_loss / len(progress_bar)

    def update_pseudo_labels(self, new_pseudo_data, new_pseudo_targets):
        """
        Update the pseudo-labeled dataset with new pseudo labels.

        Args:
            new_pseudo_data (torch.Tensor): New pseudo-labeled data
            new_pseudo_targets (torch.Tensor): New pseudo labels

        Returns:
            DataLoader: Updated pseudo-label DataLoader
        """
        if len(new_pseudo_data) == 0:
            return

        if self.pseudo_data is not None:
            self.pseudo_data = torch.cat([self.pseudo_data, new_pseudo_data], dim=0)
            self.pseudo_targets = torch.cat([self.pseudo_targets, new_pseudo_targets], dim=0)
        else:
            self.pseudo_data = new_pseudo_data
            self.pseudo_targets = new_pseudo_targets

        return DataLoader(TensorDataset(self.pseudo_data, self.pseudo_targets),
                        batch_size=self.batch_size,
                        shuffle=True)

    def generate_pseudo_labels(self, unlabeled_loader, epoch):
        """
        Generate pseudo labels for unlabeled data.

        Args:
            unlabeled_loader (DataLoader): DataLoader containing unlabeled data
            epoch (int): Current training epoch

        Returns:
            tuple: (pseudo_data, pseudo_targets, pseudo_indices)
        """
        self.model.eval()
        pseudo_data = []
        pseudo_targets = []
        selected_indices = []

        with torch.no_grad():
            progress_bar = tqdm(unlabeled_loader, desc=f"Generating Pseudo Labels (Epoch {epoch})", leave=False)

            for batch_idx, (inputs, _) in enumerate(progress_bar):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs / self.custom_loss.temperature, dim=1)
                confidence, predicted = torch.max(probabilities, dim=1)
                
                high_confidence_mask = confidence > self.confidence_threshold
                
                selected_data = inputs[high_confidence_mask]
                selected_labels = predicted[high_confidence_mask]
                
                batch_indices = torch.arange(batch_idx * self.batch_size, 
                                           batch_idx * self.batch_size + len(inputs))
                selected_batch_indices = batch_indices[high_confidence_mask.cpu()]
                
                if selected_data.size(0) > 0:
                    pseudo_data.append(selected_data)
                    pseudo_targets.append(selected_labels)
                    selected_indices.extend(selected_batch_indices.tolist())

        if pseudo_data:
            pseudo_data = torch.cat(pseudo_data, dim=0)
            pseudo_targets = torch.cat(pseudo_targets, dim=0)
            
        selected_indices = torch.tensor(selected_indices, dtype=torch.int)
        pseudo_indices = torch.tensor(unlabeled_loader.dataset.indices)[selected_indices]
        
        return pseudo_data, pseudo_targets, pseudo_indices

    def remove_pseudo_labeled_from_unlabeled(self, unlabeled_loader, pseudo_indices):
        """
        Remove pseudo-labeled samples from the unlabeled dataset.

        Args:
            unlabeled_loader (DataLoader): Current unlabeled data loader
            pseudo_indices (torch.Tensor): Indices of samples that were pseudo-labeled

        Returns:
            DataLoader: Updated unlabeled data loader
        """
        remaining_indices = list(set(unlabeled_loader.dataset.indices) - set(pseudo_indices.numpy()))
        remaining_dataset = Subset(unlabeled_loader.dataset, remaining_indices)
        remaining_dataset.indices = list(range(len(remaining_dataset)))
        return DataLoader(remaining_dataset, batch_size=self.batch_size, shuffle=True)

    def self_train(self, labeled_loader, unlabeled_loader, test_loader, num_epochs=100):
        """
        Execute the self-training loop.

        Args:
            labeled_loader (DataLoader): DataLoader for labeled training data
            unlabeled_loader (DataLoader): DataLoader for unlabeled training data
            test_loader (DataLoader): DataLoader for test data
            num_epochs (int): Number of training epochs (default: 100)

        Returns:
            tuple: (final_accuracy, class_accuracies)
        """
        total_data = len(labeled_loader.dataset) + len(unlabeled_loader.dataset)
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            
            loss = self.train_on_labeled(labeled_loader, epoch)
            labeled_accuracy = self.calculate_accuracy(labeled_loader)
            
            if epoch % 200 == 0:
                pseudo_data, pseudo_targets, pseudo_indices = self.generate_pseudo_labels(unlabeled_loader, epoch)
                
                self.pseudo_loader = self.update_pseudo_labels(pseudo_data, pseudo_targets)
                
                unlabeled_loader = self.remove_pseudo_labeled_from_unlabeled(unlabeled_loader, pseudo_indices)
                
                test_accuracy = self.calculate_accuracy(test_loader)
                
                if test_accuracy > self.best_accuracy:
                    self.best_accuracy = test_accuracy
                    self.best_model_state = self.model.state_dict().copy()
                    self.save_model()
                
                labeled_ratio = (len(labeled_loader.dataset) + 
                               (len(self.pseudo_loader.dataset) if self.pseudo_loader is not None else 0)) / total_data
                
                tqdm.write(
                    f"Epoch {epoch} Summary: "
                    f"Loss = {loss:.6f}, "
                    f"Pseudo-labels added = {len(pseudo_data)}, "
                    f"Labeled data ratio = {labeled_ratio:.2%}, "
                    f"Labeled Accuracy = {labeled_accuracy:.2f}%, "
                    f"Test Accuracy = {test_accuracy:.2f}%, "
                    f"Best Test Accuracy = {self.best_accuracy:.2f}%"
                )
            else:
                tqdm.write(f"Epoch {epoch} Summary: "
                          f"Loss = {loss:.6f}, "
                          f"Labeled Accuracy = {labeled_accuracy:.2f}%")
        
        print("\nTraining completed. Loading best model for final evaluation...")
        self.load_best_model()
        final_accuracy, class_accuracies = self.evaluate_model(test_loader)
        
        return final_accuracy, class_accuracies
