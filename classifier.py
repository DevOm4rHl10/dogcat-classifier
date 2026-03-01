# classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # skip truncated images


# =====================================================
# Device (only used during training or inference)
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================
# CNN Model Definition
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
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# =====================================================
# Safe ImageFolder (skips corrupted images)
# =====================================================
class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except Exception as e:
            print(f"Skipping corrupted image: {path} ({e})")
            return self.__getitem__((index + 1) % len(self.samples))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


# =====================================================
# Training code (runs only if this file is executed directly)
# =====================================================
if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import time, os

    # ====== Data Loading ======
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_dataset = SafeImageFolder("data/train", transform=transform)
    val_dataset = SafeImageFolder("data/val", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # ====== Model, Loss, Optimizer ======
    model = DogCatCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ====== Evaluation Function ======
    def evaluate(model, loader):
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return total_loss / len(loader), 100 * correct / total

    # ====== Training Loop ======
    def train_model(model, train_loader, val_loader, epochs=15):
        for epoch in range(epochs):
            start_time = time.time()
            model.train()
            running_loss, correct, total = 0.0, 0, 0

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
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                if batch_idx % 20 == 0:
                    elapsed = time.time() - start_time
                    avg_loss = running_loss / (batch_idx + 1)
                    acc = 100 * correct / total
                    print(f"Batch [{batch_idx}/{len(train_loader)}] | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | Time: {elapsed:.1f}s", flush=True)

            epoch_time = time.time() - start_time
            train_loss, train_acc = running_loss / len(train_loader), 100 * correct / total
            val_loss, val_acc = evaluate(model, val_loader)

            print("\n--- Epoch Summary ---")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc:.2f}%")
            print(f"Epoch Time: {epoch_time:.1f} seconds | Estimated Time Left: {epoch_time*(epochs-epoch-1)/60:.2f} min")

    # ====== Start Training ======
    print("Starting training...", flush=True)
    train_model(model, train_loader, val_loader, epochs=15)

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/dogcat_cnn.pth")
    print("\nModel saved as saved_models/dogcat_cnn.pth")