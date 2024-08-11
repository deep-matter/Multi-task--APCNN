import yaml
import torch.nn as nn

class CNN(nn.Module):
    """
    CNN is a Convolutional Neural Network (CNN model. 
    This model is designed for feature extraction.

    Attributes:
    ----------
    config : dict
        A dictionary containing all configuration parameters loaded 
        from a YAML file.
    features : nn.Sequential
        A sequential container for the convolutional and pooling 
        layers of the network.
    classifier : nn.Sequential
        A sequential container for the fully connected layers used 
        for classification.
    """
    def __init__(self, config_path="config.yml"):
        """
        Initializes the BaseCNN model with the given configuration.

        Parameters:
        ----------
        config_path : str, optional
            The path to the YAML configuration file. Default is "config.yml".
        """
        super(CNN, self).__init__()

        # Load configurations from YAML file
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)['model']

        self.features = nn.Sequential(
            nn.Conv2d(config['input_channels'], config['conv1_out_channels'],
                      kernel_size=config['conv1_kernel_size'], 
                      stride=config['conv1_stride'], 
                      padding=config['conv1_padding']),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(config['conv1_out_channels'], config['conv2_out_channels'], 
                      kernel_size=config['conv2_kernel_size'], 
                      padding=config['conv2_padding']),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(config['conv2_out_channels'], config['conv3_out_channels'], 
                      kernel_size=config['conv3_kernel_size'], 
                      padding=config['conv3_padding']),
            nn.ReLU(inplace=True),
            nn.Conv2d(config['conv3_out_channels'], config['conv4_out_channels'], 
                      kernel_size=config['conv4_kernel_size'], 
                      padding=config['conv4_padding']),
            nn.ReLU(inplace=True),
            nn.Conv2d(config['conv4_out_channels'], config['conv5_out_channels'], 
                      kernel_size=config['conv5_kernel_size'], 
                      padding=config['conv5_padding']),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(config['linear_input_dim'], config['linear1_out_dim']),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(config['linear1_out_dim'], config['linear2_out_dim']),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Defines the forward pass of the BaseCNN model.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor representing the batch of images.

        Returns:
        -------
        torch.Tensor
            Output tensor after passing through the feature extraction 
            and classification layers.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x
