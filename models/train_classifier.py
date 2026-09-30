"""
Reference:
https://github.com/pytorch/tutorials/blob/main/beginner_source/basics/quickstart_tutorial.py
"""
from binary_classifier import BinaryClassifier

import torch
from torch.utils.data import TensorDataset, DataLoader


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            predicted_labels = pred >= 0
            correct += (predicted_labels == y.bool()).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    batch_size = 10
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    # 100 random RGB images and random binary labels.
    training_data = TensorDataset(
        torch.rand(100, 3, 224, 224),
        torch.randint(0, 2, (100, 1)).float(),
    )

    test_data = TensorDataset(
        torch.rand(20, 3, 224, 224),
        torch.randint(0, 2, (20, 1)).float(),
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    model = BinaryClassifier().to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

    train(train_dataloader, model, loss_fn, optimizer, device)
    test(test_dataloader, model, loss_fn, device)


