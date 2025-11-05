from pathlib import Path

import torch
from PIL import UnidentifiedImageError
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

patience=3
script_dir = Path(__file__).resolve().parent
loss_curve_path = script_dir / "loss_curve.png"
predictions_path = script_dir / "predictions_sample.png"
model_path = script_dir / "model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except (UnidentifiedImageError, OSError):
            return self.__getitem__((index + 1) % len(self))

def _load_training_data():
    full_dataset = SafeImageFolder(root="data/PetImages", transform=None)
    train_size = int(len(full_dataset) * 0.8)
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, full_dataset.classes

def _compute_validation_loss(model, val_loader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            total_loss += loss.item() * batch_X.size(0)
    return total_loss / len(val_loader.dataset)

def _train_model(model, train_loader, val_loader):
    trigger_times = 0
    best_val_loss = float('inf')
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    for epoch in range(5):
        model.train()
        running_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)

        val_loss = _compute_validation_loss(model, val_loader, loss_fn)
        epoch_loss = running_loss / len(train_loader)
        val_losses.append(val_loss)
        train_losses.append(epoch_loss)
        model.train()

        if val_loss < best_val_loss:
            trigger_times = 0
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch + 1}: train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    _plot_loss_curve(train_losses, val_losses)


def _predict(model, data_loader):
    predictions, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1)
            predictions.append(preds.cpu())
            labels.append(batch_y.cpu())

    predictions = torch.cat(predictions)
    labels = torch.cat(labels)
    return predictions, labels

def _plot_loss_curve(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_curve_path)
    plt.close()
    print(f"Saved loss curve to {loss_curve_path}")

def _visualize_predictions(model, val_loader, class_names):
    model.eval()
    images, preds, labels = [], [], []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            predictions = torch.argmax(outputs, dim=1)
            images.extend(batch_X.cpu())
            preds.extend(predictions.cpu())
            labels.extend(batch_y.cpu())
            if len(images) >= 8:  # just a few examples
                break

    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip([0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225])],
        std=[1/s for s in [0.229, 0.224, 0.225]]
    )

    plt.figure(figsize=(12, 6))
    for i in range(8):
        img = inv_normalize(images[i]).permute(1, 2, 0).clamp(0, 1)
        plt.subplot(2, 4, i+1)
        plt.imshow(img)
        plt.axis('off')
        color = "green" if preds[i] == labels[i] else "red"
        plt.title(f"Pred: {class_names[preds[i]]}\nTrue: {class_names[labels[i]]}",
                  color=color, fontsize=9)

    plt.tight_layout()
    plt.savefig(predictions_path)
    plt.close()
    print(f"Saved prediction samples to {predictions_path}")

def main():
    print(f"Using device: {device}")
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    #Cat/Dog
    num_classes = 2
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    train_loader, val_loader, class_names = _load_training_data()
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        model.to(device)
        _train_model(model, train_loader, val_loader)


    model.to(device)
    predictions, labels = _predict(model, val_loader)
    accuracy = (predictions == labels).float().mean().item()
    print(f"Accuracy: {accuracy:.4f}")

    _visualize_predictions(model, val_loader, class_names)

if __name__ == '__main__':
    main()
