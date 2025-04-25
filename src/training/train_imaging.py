import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import cv2
from pathlib import Path
from models.hybrid_model import HybridImagingModel

# -------------------------------
# Custom Dataset for Processed Imaging Data
# -------------------------------
class ProcessedImagingDataset(Dataset):
    """
    Custom dataset for loading preprocessed images and corresponding labels.
    Assumes images are stored in data/processed/dataset1/train/images
    and labels are stored in data/processed/dataset1/train/labels.
    
    Label extraction: reads the first number from the label file as class label.
    """
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_files = list(self.images_dir.glob("*.jpg"))
        self.transform = transform
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        # Read image using cv2 then convert to RGB
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Optionally apply transforms (e.g., ToTensor, normalization)
        if self.transform:
            img = self.transform(img)
        else:
            # Convert to float tensor and scale [0,255] to [0,1]
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Get corresponding label file: assume same name with .txt extension
        label_file = self.labels_dir / (img_path.stem + ".txt")
        # Default label is 0 if not found (shouldn't happen if preprocessing was correct)
        label = 0  
        if label_file.exists():
            with open(label_file, "r") as f:
                # Read the first line and take the first value as the class label
                line = f.readline().strip()
                if line:
                    label = int(line.split()[0])
        return img, label

# -------------------------------
# Model: Hybrid Imaging Model with Classifier Head
# -------------------------------
class ImagingClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImagingClassifier, self).__init__()
        self.hybrid_model = HybridImagingModel()  # Outputs 512-dim features
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        features = self.hybrid_model(x)  # (batch, 512)
        logits = self.classifier(features)
        return logits

# -------------------------------
# Training Function
# -------------------------------
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)
        
        epoch_loss = running_loss / total
        epoch_acc = correct.double() / total
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")
    
    return model

# -------------------------------
# Main Training Loop
# -------------------------------
if __name__ == "__main__":
    # Set directories for Dataset1 (adjust paths as needed)
    images_dir = "data/processed/dataset1/train/images"
    labels_dir = "data/processed/dataset1/train/labels"
    
    # Create dataset and dataloader
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((640, 640)),
        T.ToTensor(),  # scales [0,255] to [0,1]
    ])
    train_dataset = ProcessedImagingDataset(images_dir, labels_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    
    # Number of classes - update based on your dataset (e.g., 2 for binary or more)
    num_classes = 2  # Example: benign vs. malignant
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model, criterion, optimizer
    model = ImagingClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print("Starting training of the Imaging Module...")
    trained_model = train_model(model, train_loader, criterion, optimizer, device, num_epochs=20)
    
    # Save the model
    torch.save(trained_model.state_dict(), "trained_imaging_model.pth")
    print("Training completed and model saved as trained_imaging_model.pth")
