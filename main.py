import argparse
import yaml
import os
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score
from torch.cuda import amp
from src.models.cnn import CNN
from src.models.mtl import  MTL
from src.models.optimizer import optimize_s, optimize_l
from src.models.loss import hinge_loss
from .train import train 

def main():
    parser = argparse.ArgumentParser(description='Train Multi-Task Learning Framework for Binary Semantic Attribute Prediction')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    parser.add_argument('--data_loader', type=str, required=True, help='Path to the data loader module')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    base_cnn = CNN(config_path=args.config)

    model = MTL(base_cnn, config['model'])

    # Define  parameters for S and L
    params_s = model.combination_matrix.parameters()
    params_l = model.latent_layer.parameters()

    # Define - optimizers for S and L
    optimizer_s = optim.Adam(params_s, lr=config['model']['learning_rate_s'])
    optimizer_l = optim.Adam(params_l, lr=config['model']['learning_rate_l'])

    data_loader_module = __import__(args.data_loader, fromlist=['data_loader'])
    data_loader = data_loader_module.data_loader

    # Ensure the checkpoint directory exists
    os.makedirs(config['model']['checkpoint_path'], exist_ok=True)

    train_model(model, data_loader, optimizer_s, optimizer_l, config['model']['num_epochs'], config['model']['early_stop_patience'], config['model']['checkpoint_path'])

if __name__ == '__main__':
    main()