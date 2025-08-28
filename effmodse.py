import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Focal Loss with Label Smoothing
class FocalLossWithSmoothing(nn.Module):
    def __init__(self, alpha=1, gamma=2, smoothing=0.1):
        super(FocalLossWithSmoothing, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        smooth_targets = (
            torch.zeros_like(inputs)
            .scatter_(1, targets.unsqueeze(1), 1)
            * (1 - self.smoothing)
            + (self.smoothing / num_classes)
        )

        inputs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(inputs)

        focal_weight = self.alpha * ((1 - probs) ** self.gamma)
        loss = (-focal_weight * smooth_targets * inputs).sum(dim=1).mean()
        return loss

# Adaptive Squeeze-and-Excitation Block with Spatial Attention
class AdaptiveSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(AdaptiveSEBlock, self).__init__()

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

        # Spatial Attention
        self.spatial_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.avg_pool(x)
        avg_out = avg_out.view(avg_out.size(0), -1)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.fc2(avg_out)
        avg_out = self.sigmoid(avg_out).view(avg_out.size(0), -1, 1, 1)

        # Spatial attention
        spatial_out = self.spatial_conv(x)
        spatial_out = self.sigmoid_spatial(spatial_out)

        # Combine both attentions
        out = x * avg_out
        out = out * spatial_out
        return out

# Data Augmentation and Transformation
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Path to the dataset's 'crop' folder where the images are stored in subdirectories by class
data_dir = "D:/militaryproject/crop"  # Update this with your actual path

# Load the dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform_train)

# Split dataset into training and validation sets (80-20 split)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Update the validation dataset's transform
val_dataset.dataset.transform = transform_val

# DataLoader for training and validation
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load pre-trained EfficientNet model
model = models.efficientnet_b0(pretrained=True)

# Add Adaptive SE block after the feature extractor layers
class EfficientNetWithSE(nn.Module):
    def __init__(self, model):
        super(EfficientNetWithSE, self).__init__()
        self.features = model.features  # feature extractor layers
        self.se_block = AdaptiveSEBlock(1280)  # In EfficientNet-B0, the number of channels is 1280
        self.classifier = model.classifier  # The final classifier layers
        
    def forward(self, x):
        x = self.features(x)
        x = self.se_block(x)  # Apply the SE block
        x = x.mean([2, 3])  # Global average pooling
        x = self.classifier(x)  # Final classification
        return x

# Create model with SE block
model = EfficientNetWithSE(model)

# Move model to device
model.to(device)

# Loss function and optimizer
criterion = FocalLossWithSmoothing(alpha=1, gamma=2, smoothing=0.1)  # Focal Loss with Label Smoothing
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and validation functions
def train(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Train and validate the model
num_epochs = 20
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}:")
    
    # Training phase
    train_loss, train_acc = train(model, train_loader, criterion, optimizer)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

    # Validation phase
    val_loss, val_acc = validate(model, val_loader, criterion)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

# Plotting training and validation loss and accuracy
# Plot Losses
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracies
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
torch.save(model.state_dict(), "efficientnet_model_with_adaptive_se.pth")
print("Model saved as 'efficientnet_model_with_adaptive_se.pth'")
