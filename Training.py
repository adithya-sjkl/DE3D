import torch
import torch.nn as nn
import torchvision
from accelerate import Accelerator
import matplotlib.pyplot as plt
import torch.optim as optim
from accelerate.utils import set_seed
from Efficient3DModel import DE3D
from Preprocessing import prepare
from sklearn.metrics import accuracy_score, f1_score

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

def training_loop(mixed_precision="fp16", seed:int=42, batch_size:int=1, num_epochs=10, healthy_dir, disease_dir, int_channels:int=10):
    set_seed(seed)
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision=mixed_precision)
    # Build dataloaders
    train_dataloader, val_dataloader, test_dataloader = prepare(healthy_dir=healthy_dir,disease_dir=disease_dir,batch_size=batch_size)

    # Build model
    model = DE3D(channels=int_channels,batch_size=batch_size,feat_res=28)

    optim = optim.Adam(model.parameters())

    scheduler = torch.optim.lr_scheduler.ConstantLR(optim,last_epoch=-1)
    model,optimizer,train_dataloader,scheduler  = accelerator.prepare(model,optim,train_dataloader,scheduler )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Training loop
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.logits
            loss = criterion(outputs, labels)
            accelerator.backward(loss)
            optimizer.step()

            running_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

        average_loss = running_loss / len(train_dataloader)
        training_accuracy = total_correct / total_samples
        train_losses.append(average_loss)
        train_accuracies.append(training_accuracy)

        accelerator.print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_loss:.4f}, Training Accuracy: {training_accuracy:.4f}")

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(accelerator.device), labels.to(accelerator.device)
                outputs = model(inputs)
                outputs = outputs.logits
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.to(accelerator.device)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_running_loss += loss.item()

        average_val_loss = val_running_loss / len(val_dataloader)
        val_accuracy = val_correct / val_total
        accelerator.print(f"Validation Loss: {average_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        val_losses.append(average_val_loss)
        val_accuracies.append(val_accuracy)
    
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = model(inputs).squeeze()
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = accuracy_score(all_labels, all_preds)
    test_f1_score = f1_score(all_labels, all_preds)
    print(f"Test Accuracy: {test_accuracy:.4f}, Test F1 Score: {test_f1_score:.4f}")
    print("Training complete!")


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
