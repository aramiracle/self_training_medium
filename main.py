import torch
import torch.optim as optim
import numpy as np
import random
from preprocess import prepare_data
from model import create_model
from trainer import SelfTraining

def main():
    batch_size = 1024
    num_epochs = 1000
    learning_rate = 1e-4
    confidence_threshold = 0.95
    pseudo_weight = 0.2
    labeled_ratio = 0.01
    seed = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    labeled_loader, unlabeled_loader, test_loader = prepare_data(labeled_ratio, batch_size)

    model = create_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    trainer = SelfTraining(
        model=model,
        optimizer=optimizer,
        confidence_threshold=confidence_threshold,
        batch_size=batch_size,
        device=device,
        pseudo_weight=pseudo_weight,
        model_save_path='best_mnist_model.pth'  # Optional: specify custom save path
    )
    final_accuracy, _ = trainer.self_train(labeled_loader, unlabeled_loader, test_loader, num_epochs)

    print("\nTraining Complete!")
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")

    # Access the pseudo-labeled DataLoader
    pseudo_loader = trainer.pseudo_loader
    print(f"Number of pseudo-labeled samples: {len(pseudo_loader.dataset)}")


if __name__ == "__main__":
    main()
