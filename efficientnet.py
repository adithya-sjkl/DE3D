import torch

from efficientnet_pytorch import EfficientNet
import torch

# Load EfficientNet (e.g., EfficientNet-B0)
efficientnet = EfficientNet.from_pretrained('efficientnet-b0')

class FeatureExtractor(nn.Module):
    def __init__(self, efficientnet):
        super(FeatureExtractor, self).__init__()
        self.efficientnet = efficientnet
        self.feature_blocks = nn.Sequential(*list(efficientnet.children())[:-2])  # Extract features after the last block

    def forward(self, x):
        return self.feature_blocks(x)
    
    image = torch.randn(224,224,3)

    print(image)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Define a simple binary classification model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Generate dummy data
def generate_dummy_data():
    np.random.seed(0)
    X_train = np.random.rand(1000, 10)
    y_train = (np.random.rand(1000) > 0.5).astype(np.float32)
    X_val = np.random.rand(200, 10)
    y_val = (np.random.rand(200) > 0.5).astype(np.float32)
    X_test = np.random.rand(200, 10)
    y_test = (np.random.rand(200) > 0.5).astype(np.float32)

    return X_train, y_train, X_val, y_val, X_test, y_test

# Load data
X_train, y_train, X_val, y_val, X_test, y_test = generate_dummy_data()
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Initialize model, loss function, and optimizer
model = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    train_loss = running_loss / total
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    val_loss = running_loss / total
    val_accuracy = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Test the model
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs).squeeze()
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = accuracy_score(all_labels, all_preds)
test_f1_score = f1_score(all_labels, all_preds)
print(f"Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1_score:.4f}")

# Plot the training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Losses')
plt.savefig('losses.png')

# Plot the training and validation accuracies
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracies')
plt.savefig('accuracies.png')

# Save the model
torch.save(model.state_dict(), 'model.pth')

# Save metrics
with open('metrics.txt', 'w') as f:
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test F1 Score: {test_f1_score:.4f}\n")
