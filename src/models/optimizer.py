import torch
import torch.nn.functional as F
from torch.nn import optim
from torch.cuda import amp

def hinge_loss(outputs, targets):
    """
    Compute the hinge loss.

    Parameters:
    ----------
    outputs : torch.Tensor
        The model's predicted outputs.
    targets : torch.Tensor
        The ground truth labels.

    Returns:
    -------
    torch.Tensor
        The computed hinge loss.
    """
    return torch.mean(torch.clamp(1 - outputs * targets, min=0))


def optimize_s_step(model, inputs, attributes, optimizer_s, scaler):
    """
    Perform an optimization step for the task-specific combination matrices S.

    Parameters:
    ----------
    model : MTL
        The multitask learning model.
    inputs : torch.Tensor
        The input tensor representing the batch of images.
    attributes : torch.Tensor
        The ground truth labels for the binary attributes.
    optimizer_s : torch.optim.Optimizer
        The optimizer for the task-specific layers (S matrices).
    scaler : torch.cuda.amp.GradScaler
        The scaler for mixed precision training.
    """
    model.train()
    optimizer_s.zero_grad()
    with amp.autocast():
        predictions = model(inputs)
        loss = hinge_loss(predictions, attributes)
    scaler.scale(loss).backward(retain_graph=True)
    scaler.step(optimizer_s)
    scaler.update()

def optimize_l_step(model, inputs, attributes, optimizer_l, scaler):
    """
    Perform an optimization step for the shared latent task matrix L.

    Parameters:
    ----------
    model : MTL
        The multitask learning model.
    inputs : torch.Tensor
        The input tensor representing the batch of images.
    attributes : torch.Tensor
        The ground truth labels for the binary attributes.
    optimizer_l : torch.optim.Optimizer
        The optimizer for the shared latent task matrix (L).
    scaler : torch.cuda.amp.GradScaler
        The scaler for mixed precision training.
    """
    model.train()
    optimizer_l.zero_grad()
    with amp.autocast():
        predictions = model(inputs)
        loss = hinge_loss(predictions, attributes)
    scaler.scale(loss).backward()
    scaler.step(optimizer_l)
    scaler.update()

def optimize_model(model, inputs, attributes, optimizer_s, optimizer_l, scaler, num_iterations):
    """
    Optimize the model by alternating between S and L optimization steps.

    Parameters:
    ----------
    model : MTL
        The multitask learning model.
    inputs : torch.Tensor
        The input tensor representing the batch of images.
    attributes : torch.Tensor
        The ground truth labels for the binary attributes.
    optimizer_s : torch.optim.Optimizer
        The optimizer for the task-specific layers (S matrices).
    optimizer_l : torch.optim.Optimizer
        The optimizer for the shared latent task matrix (L).
    scaler : torch.cuda.amp.GradScaler
        The scaler for mixed precision training.
    num_iterations : int
        The number of optimization iterations to perform.
    """
    for iteration in range(num_iterations):
        optimize_s_step(model, inputs, attributes, optimizer_s, scaler)
        optimize_l_step(model, inputs, attributes, optimizer_l, scaler)