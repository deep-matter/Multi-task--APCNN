import yaml
import torch.nn as nn

class MTL(nn.Module):
    """
    Multitask Learning (MTL) Framework for binary semantic attribute prediction.

    This framework utilizes a shared latent task matrix (L) and task-specific combination
    matrices (S) for each attribute. The model is trained using custom optimization routines.

    Attributes:
    ----------
    base_cnn : nn.Module
        The base convolutional neural network used for feature extraction.
    latent_task_matrix : nn.Linear
        The shared latent task matrix (L) for feature transformation.
    task_specific_layers : nn.ModuleList
        Task-specific combination matrices (S) for each attribute.
    """
    def __init__(self, base_cnn, num_attributes):
        """
        Initialize the MTLFramework.

        Parameters:
        ----------
        base_cnn : nn.Module
            The base CNN model.
        num_attributes : int
            The number of binary attributes to predict.
        """
        super(MTL, self).__init__()
        self.base_cnn = base_cnn
        
        # Shared latent task matrix L
        self.latent_task_matrix = nn.Linear(4096, 1024)  # Reduced dimensionality for shared latent features
        
        # Task-specific combination matrices S for each attribute
        self.task_specific_layers = nn.ModuleList([nn.Linear(1024, 1) for _ in range(num_attributes)])
    
    def forward(self, x):
        """
        Forward pass of the MTLFramework.

        Parameters:
        ----------
        x : torch.Tensor
            The input tensor representing a batch of images.

        Returns:
        -------
        torch.Tensor
            The combined outputs from all tasks.
        """
        x = self.base_cnn(x)  # Pass input through the base CNN
        x = self.latent_task_matrix(x)  # Shared latent features
        
        # Predict binary attributes
        outputs = []
        for task_layer in self.task_specific_layers:
            outputs.append(task_layer(x))
        
        return torch.cat(outputs, dim=1)  # Combine outputs from all tasks
