import torch
import matplotlib.pyplot as plt
from typing import List

def plot_3x3(
        dataloader: torch.utils.data.DataLoader, 
        classes: List[str] = None
    ) -> None:
    images, labels = next(iter(dataloader))
    
    _, axes = plt.subplots(3, 3, figsize=(4, 4))
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].numpy().transpose((1, 2, 0)))
        ax.set_title(classes[labels[i].item()])
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()

def plot_loss_accuracy(
        train_losses: List[int], 
        test_losses: List[int], 
        train_accuracies: List[int], 
        test_accuracies: List[int],
    ) -> None:
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training loss")
    plt.plot(epochs, test_losses, label="Testing loss")
    plt.title("Loss Plot")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training accuracy")
    plt.plot(epochs, test_accuracies, label="Testing accuracy")
    plt.title("Accuracy Plot")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()
