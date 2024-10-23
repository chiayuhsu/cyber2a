import torch
from tqdm import tqdm

def train(model, criterion, optimizer, train_loader, val_loader, num_epochs=5):
    """
    Train the model.
    
    Args:
        model: The model to train.
        criterion: The loss function.
        optimizer: The optimizer.
        train_loader: DataLoader for the training data.
        val_loader: DataLoader for the validation data.
        num_epochs (int): Number of epochs to train.
    
    Returns:
        model: The trained model.
    """
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        running_loss = 0.0
        for i, data in enumerate(tqdm(train_loader)):
            inputs, labels = data
            
            # get model's device
            device = next(model.parameters()).device
            
            # Move data to the appropriate device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass to get model outputs
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)
            # Backward pass to compute gradients
            loss.backward()
            # Update model parameters
            optimizer.step()

            # Accumulate the running loss
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}")
        
        # Validation phase
        # set the model to validation mode
        model.eval()
        correct = 0
        total = 0
        # Disable gradient computation for validation
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                # Move validation data to the appropriate device
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # Get the predicted class
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Validation accuracy: {100 * correct / total}%")

    return model