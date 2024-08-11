import torch
import torch.nn as nn

class MTL(nn.Module):
    """
    Multitask Learning (MTL) Framework for binary semantic attribute prediction.

    This framework utilizes a shared latent task matrix (L) and task-specific combination
    matrices (S) for each attribute. Multiple CNNs are used in parallel to extract features.
    
    Attributes:
    ----------
    base_cnns : nn.ModuleList
        A list of base convolutional neural networks used for feature extraction from different groups.
    latent_task_matrix : nn.Linear
        The shared latent task matrix (L) for feature transformation.
    task_specific_layers : nn.ModuleList
        Task-specific combination matrices (S) for each attribute.
    """
    def __init__(self, base_cnns, num_attributes, latent_dim=1024):
        """
        Initialize the MTLFramework.

        Parameters:
        ----------
        base_cnns : list of nn.Module
            A list of base CNN models, each representing a group.
        num_attributes : int
            The number of binary attributes to predict.
        latent_dim : int, optional
            The dimension of the shared latent task matrix. Default is 1024.
        """
        super(MTL, self).__init__()
        self.base_cnns = nn.ModuleList(base_cnns)
        
        # Shared latent task matrix L
        self.latent_task_matrix = nn.Linear(4096, latent_dim)  # Reduced dimensionality for shared latent features
        
        # Task-specific combination matrices S for each attribute
        self.task_specific_layers = nn.ModuleList([nn.Linear(latent_dim, 1) for _ in range(num_attributes)])
    
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
        # Apply each CNN to the input
        group_features = [cnn(x) for cnn in self.base_cnns]
        
        # Concatenate all group features (assuming they are compatible in size)
        x = torch.cat(group_features, dim=1)
        
        # Apply the shared latent task matrix (L)
        x = self.latent_task_matrix(x)
        
        # Predict binary attributes using task-specific layers (S)
        outputs = []
        for task_layer in self.task_specific_layers:
            outputs.append(task_layer(x))
        
        return torch.cat(outputs, dim=1)  