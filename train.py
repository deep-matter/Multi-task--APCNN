import os
from sklearn.metrics import accuracy_score, recall_score
from .src.models import *
def train_model(model, data_loader, optimizer_s, optimizer_l, num_epochs, early_stop_patience, checkpoint_path):
    """
    Train the MTLFramework model with custom optimization routines for S and L matrices,
    incorporating FP16 autocast, early stopping, checkpoint saving, and accuracy/recall computation.

    Parameters:
    ----------
    model : MTLFramework
        The multitask learning model.
    data_loader : torch.utils.data.DataLoader
        DataLoader providing the training data.
    optimizer_s : torch.optim.Optimizer
        The optimizer for the S matrix.
    optimizer_l : torch.optim.Optimizer
        The optimizer for the L matrix.
    num_epochs : int
        The number of training epochs.
    early_stop_patience : int
        Number of epochs to wait for improvement before early stopping.
    checkpoint_path : str
        Directory to save the model checkpoints.

    Returns:
    -------
    None
    """
    scaler = amp.GradScaler()  # For mixed precision training
    best_loss = float('inf')
    epochs_no_improve = 0



    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        all_targets = []
        all_predictions = []

        for images, labels in data_loader:
            # Optimize S
            optimize_s(model, images, labels, optimizer_s, scaler)
            
            # Optimize L
            optimize_l(model, images, labels, optimizer_l, scaler)
            
            # Collect predictions and labels for metrics computation
            with amp.autocast():
                predictions = model(images)
                loss = hinge_loss(predictions, labels)
                epoch_loss += loss.item()
                all_targets.extend(labels.cpu().numpy())
                all_predictions.extend(torch.sign(predictions).cpu().numpy())  # Binarize predictions

        accuracy = accuracy_score(all_targets, all_predictions)
        recall = recall_score(all_targets, all_predictions, average='macro')

        # Logging
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}')

        # Early Stopping and Model Checkpointing
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
            # Save model checkpoint
            torch.save(model.state_dict(), os.path.join(checkpoint_path, f'best_model_epoch_{epoch+1}.pt'))
            print(f"Model checkpoint saved at epoch {epoch+1}.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break