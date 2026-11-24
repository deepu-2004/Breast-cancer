import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader
from collections import Counter
import os

# ================= PATHS =================
TRAIN_DIR = os.path.join("Data", "train")
TEST_DIR = os.path.join("Data", "test")
MODEL_DIR = "models"
EPOCHS = 5
BATCH_SIZE = 16
# ========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(MODEL_DIR, exist_ok=True)

# -------- TRANSFORMS --------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------- DATASETS --------
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

print("Classes:", train_dataset.classes)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------- CLASS WEIGHTS --------
labels = [label for _, label in train_dataset.samples]
counts = Counter(labels)
weights = [1.0 / counts[i] for i in range(len(counts))]
class_weights = torch.tensor(weights).to(device)

# -------- MODEL --------
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.last_channel, len(train_dataset.classes))
model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_acc = 0

# -------- TRAINING LOOP --------
for epoch in range(EPOCHS):
    model.train()
    correct = total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    # -------- VALIDATION --------
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}"
    )

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(
            model.state_dict(),
            os.path.join(MODEL_DIR, "mobilenetv2_best.pt")
        )

print("\n Training complete. Best model saved.")
