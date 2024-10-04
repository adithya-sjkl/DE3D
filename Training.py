import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,f1_score
import numpy as np
from torchvision.models import resnet18
from torch import optim
from Model import E3D
from Preprocessing import prepare

def Train(healthy_dir, disease_dir, rmin:float, rmax:float, batch_size:int=3, num_epochs:int=10, lr:float=0.001, num_slices:int=7, dropout:float=0.5):
    print('change test')

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Training parameters
    num_epochs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    #Define model and dataloaders
    model = E3D(batch_size=batch_size, num_slices=num_slices, rmin=rmin, rmax=rmax, dropout=dropout).to(device)

    train_loader, val_loader, test_loader = prepare(healthy_dir, disease_dir, batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(),lr=lr)

    # Lists to store metrics for plotting
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    '''
    def calculate_accuracy(y_pred, y_true):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))
        correct_results_sum = (y_pred_tag == y_true).sum().float()
        acc = correct_results_sum / y_true.shape[0]
        return acc.item()
    '''
    def calculate_accuracy(outputs, labels):
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        return correct / total

    def train_epoch(model, dataloader, optimizer, criterion, device):
        torch.autograd.set_detect_anomaly(True)
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
        for i, (inputs, labels) in pbar:
            labels = labels.to(torch.long)
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward(retain_graph=True)
            optimizer.step()
            
            running_loss += loss.item()
            running_acc += calculate_accuracy(outputs.squeeze(), labels)
            
            pbar.set_postfix({'loss': running_loss / (i+1), 'acc': running_acc / (i+1)})
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = running_acc / len(dataloader)
        return epoch_loss, epoch_acc

    def validate_epoch(model, dataloader, criterion, device):
        model.eval()
        running_loss = 0.0
        running_acc = 0.0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                labels = labels.to(torch.long)
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                
                running_loss += loss.item()
                running_acc += calculate_accuracy(outputs.squeeze(), labels)
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = running_acc / len(dataloader)
        return epoch_loss, epoch_acc

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Test the model
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # Get the index of the max log-probability
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)

    # Calculate accuracy
    test_acc = accuracy_score(test_labels, test_preds)

    # Calculate F1 score (you can choose 'micro', 'macro', or 'weighted' average)
    test_f1 = f1_score(test_labels, test_preds, average='weighted')

    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score (weighted): {test_f1:.4f}")

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Save the model
    torch.save(model.state_dict(), 'binary_classification_model.pth')
    print("Model saved successfully.")


''' 
healthy_dir = "/Users/adithyasjith/Documents/Code/DE3D/Data/NC"
disease_dir = "/Users/adithyasjith/Documents/Code/DE3D/Data/AD"


#Hyperparameters
rmin=0.5
rmax=0.99
batch_size=3
num_epochs=10
lr=0.001
num_slices=7
dropout=0.5
 
Train(healthy_dir, disease_dir, rmin, rmax, batch_size, num_epochs, lr, num_slices, dropout)

'''