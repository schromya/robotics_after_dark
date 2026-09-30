"""
Reference:
https://github.com/pytorch/tutorials/blob/main/beginner_source/basics/quickstart_tutorial.py
"""
from binary_classifier import BinaryClassifier

from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def load_data(data_dir, batch_size=10):
    """
    Load labeled images and create a reproducible 80/20 train/test split.
    Args:
        data_dir: Directory containing failure/ and success/ image folders.
        batch_size: Number of images in each batch.
    Returns:
        Training and test DataLoaders with images resized to 224 x 224.
    """

    # TODO (schromya): Crop image to 224x224 to begin with?
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=preprocess)
    if dataset.class_to_idx != {"failure": 0, "success": 1}:
        raise ValueError("Expected only failure/ and success/ folders in the dataset.")
    if len(dataset) < 2:
        raise ValueError("Need at least two images for a train/test split.")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    training_data, test_data = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False
    )
    return train_dataloader, test_dataloader


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    model.backbone.eval()  # Keep the frozen backbone's BatchNorm statistics fixed.
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device).float().reshape(-1, 1)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = loss.item(), min((batch + 1) * dataloader.batch_size, size)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device).float().reshape(-1, 1)
            pred = model(X)
            test_loss += loss_fn(pred, y).item() * X.size(0)
            predicted_labels = pred >= 0
            correct += (predicted_labels == y.bool()).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    batch_size = 10
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    data_dir = Path(__file__).resolve().parents[1] / "data" / "classifier"

    train_dataloader, test_dataloader = load_data(data_dir, batch_size)

    model = BinaryClassifier().to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-4)

    train(train_dataloader, model, loss_fn, optimizer, device)
    test(test_dataloader, model, loss_fn, device)


