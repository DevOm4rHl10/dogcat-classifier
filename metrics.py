import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from classifier import DogCatCNN, SafeImageFolder


import warnings
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")



# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load model
model = DogCatCNN().to(device)
model.load_state_dict(torch.load("saved_models/dogcat_cnn.pth", map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Test dataset
print("Loading test dataset...")
test_dataset = SafeImageFolder("data/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print("Number of test batches:", len(test_loader))

# Collect predictions and labels
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).long().squeeze(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()



TN, FP, FN, TP = cm.ravel()  # flatten 2x2 matrix

print(TN, FP, FN, TP)


# Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)
print(f"Accuracy: {accuracy:.4f}")

# Precision
precision_dog = TP / (TP + FP)  # Predicted dogs that are actually dogs
precision_cat = TN / (TN + FN)  # Predicted cats that are actually cats
print(f"Precision - Dog: {precision_dog:.4f}")
print(f"Precision - Cat: {precision_cat:.4f}")

# Recall
recall_dog = TP / (TP + FN)  # Actual dogs correctly predicted
recall_cat = TN / (TN + FP)  # Actual cats correctly predicted
print(f"Recall - Dog: {recall_dog:.4f}")
print(f"Recall - Cat: {recall_cat:.4f}")

# F1 Score
f1_dog = 2 * (precision_dog * recall_dog) / (precision_dog + recall_dog)
f1_cat = 2 * (precision_cat * recall_cat) / (precision_cat + recall_cat)
print(f"F1 Score - Dog: {f1_dog:.4f}")
print(f"F1 Score - Cat: {f1_cat:.4f}")

