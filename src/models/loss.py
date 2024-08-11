import torch.nn.functional as F
import torch.optim as optim

class HingeLoss(nn.Module):
    """
    HingeLoss computes the hinge loss, which is often used in binary classification tasks,
    especially in Support Vector Machines (SVMs). The hinge loss is defined as:
    
    L(y, f(x)) = max(0, 1 - y * f(x))
    
    Attributes:
    ----------
    None
    """
    def __init__(self):
        """
        Initializes the HingeLoss class.
        """
        super(HingeLoss, self).__init__()
    
    def forward(self, outputs, targets):
        """
        Computes the forward pass of the hinge loss.
        
        Parameters:
        ----------
        outputs : torch.Tensor
            The predicted outputs from the model.
        targets : torch.Tensor
            The ground truth labels for the corresponding inputs.
        
        Returns:
        -------
        torch.Tensor
            The computed hinge loss.
        """
        hinge_loss = torch.mean(torch.clamp(1 - outputs * targets, min=0))
        return hinge_loss



