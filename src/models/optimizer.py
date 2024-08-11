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


def optimize_s(model, images, labels, optimizer_s, scaler):
    # SPGD optimization for S
    optimizer_s.zero_grad()
    with amp.autocast():
        predictions = model(images)
        loss = hinge_loss(predictions, labels)
    scaler.scale(loss).backward(retain_graph=True)  
    # Retain the graph to use gradients for L optimization
    scaler.step(optimizer_s)
    scaler.update()

def optimize_l(model, images, labels, optimizer_l, scaler):
    #  APG optimization for L
    optimizer_l.zero_grad()
    with amp.autocast():
        predictions = model(images)
        loss = hinge_loss(predictions, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer_l)
    scaler.update()