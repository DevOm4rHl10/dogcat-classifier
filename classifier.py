import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import time
import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



# =====================================================
# Device
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# Model Definition
# =====================================================
class DogCatCNN(nn.Module):
    def __init__(self):
        super(DogCatCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 128 -> 64
        x = self.pool(F.relu(self.conv2(x)))  # 64 -> 32
        x = self.pool(F.relu(self.conv3(x)))  # 32 -> 16

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x



"""
For  ensuring image is not corrupted : 
"""
class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]

        # Try loading the image, skip if fails
        try:
            sample = self.loader(path)
        except Exception as e:
            print(f"Skipping corrupted image: {path} ({e})")
            # Pick a random valid image instead
            return self.__getitem__((index + 1) % len(self.samples))

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target



# =====================================================
# Data Loading
# =====================================================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

print("Loading datasets...")

train_dataset = SafeImageFolder("data/train", transform=transform)
val_dataset = SafeImageFolder("data/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

print("Number of training batches:", len(train_loader))


# =====================================================
# Model, Loss, Optimizer
# =====================================================
model = DogCatCNN().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# =====================================================
# Evaluation Function
# =====================================================
def evaluate(model, loader):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


# =====================================================
# Training Function
# =====================================================
def train_model(model, train_loader, val_loader, epochs=15):

    for epoch in range(epochs):

        start_time = time.time()
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\n===== Epoch {epoch+1}/{epochs} =====", flush=True)

        for batch_idx, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Print every 20 batches
            if batch_idx % 20 == 0:
                elapsed = time.time() - start_time
                avg_loss = running_loss / (batch_idx + 1)
                acc = 100 * correct / total

                print(
                    f"Batch [{batch_idx}/{len(train_loader)}] | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Acc: {acc:.2f}% | "
                    f"Time: {elapsed:.1f}s",
                    flush=True
                )

        # Epoch Summary
        epoch_time = time.time() - start_time
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        val_loss, val_acc = evaluate(model, val_loader)

        print("\n--- Epoch Summary ---")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Acc : {train_acc:.2f}%")
        print(f"Val Loss  : {val_loss:.4f}")
        print(f"Val Acc   : {val_acc:.2f}%")
        print(f"Epoch Time: {epoch_time:.1f} seconds")

        remaining_time = epoch_time * (epochs - epoch - 1)
        print(f"Estimated Time Left: {remaining_time/60:.2f} minutes")


# =====================================================
# Run Training
# =====================================================
if __name__ == "__main__":

    print("Starting training...", flush=True)

    train_model(model, train_loader, val_loader, epochs=15)

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/dogcat_cnn.pth")

    print("\nModel saved as saved_models/dogcat_cnn.pth")