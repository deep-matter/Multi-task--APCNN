mport torch
import torch.nn as nn

class CNN(nn.Module):
    """
    CNN is a Convolutional Neural Network (CNN) model. 
    This model is designed for feature extraction.

    Attributes:
    ----------
    features : nn.Sequential
        A sequential container for the convolutional and pooling 
        layers of the network.
    classifier : nn.Sequential
        A sequential container for the fully connected layers used 
        for classification.
    """
    def __init__(self, input_channels=3, conv1_out_channels=64, conv1_kernel_size=11, 
                 conv1_stride=4, conv2_out_channels=192, conv2_kernel_size=5, 
                 conv3_out_channels=384, conv3_kernel_size=3, conv4_out_channels=256, 
                 conv4_kernel_size=3, conv5_out_channels=256, conv5_kernel_size=3, 
                 linear_input_dim=256*6*6, linear1_out_dim=4096, linear2_out_dim=4096):
        """
        Initializes the CNN model with the given parameters.

        Parameters:
        ----------
        input_channels : int
            Number of input channels (e.g., 3 for RGB images).
        conv1_out_channels : int
            Number of output channels for the first convolutional layer.
        conv1_kernel_size : int
            Kernel size for the first convolutional layer.
        conv1_stride : int
            Stride for the first convolutional layer.
        conv2_out_channels : int
            Number of output channels for the second convolutional layer.
        conv2_kernel_size : int
            Kernel size for the second convolutional layer.
        conv3_out_channels : int
            Number of output channels for the third convolutional layer.
        conv3_kernel_size : int
            Kernel size for the third convolutional layer.
        conv4_out_channels : int
            Number of output channels for the fourth convolutional layer.
        conv4_kernel_size : int
            Kernel size for the fourth convolutional layer.
        conv5_out_channels : int
            Number of output channels for the fifth convolutional layer.
        conv5_kernel_size : int
            Kernel size for the fifth convolutional layer.
        linear_input_dim : int
            Input dimension for the first fully connected layer.
        linear1_out_dim : int
            Output dimension for the first fully connected layer.
        linear2_out_dim : int
            Output dimension for the second fully connected layer.
        """
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, conv1_out_channels, 
                      kernel_size=conv1_kernel_size, 
                      stride=conv1_stride),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(conv1_out_channels, conv2_out_channels, 
                      kernel_size=conv2_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(conv2_out_channels, conv3_out_channels, 
                      kernel_size=conv3_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv3_out_channels, conv4_out_channels, 
                      kernel_size=conv4_kernel_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv4_out_channels, conv5_out_channels, 
                      kernel_size=conv5_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(linear_input_dim, linear1_out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(linear1_out_dim, linear2_out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Defines the forward pass of the CNN model.

        Parameters:
        ----------
        x : torch.Tensor
            Input tensor representing the batch of images.

        Returns:
        -------
        torch.Tensor
            Output tensor after passing through the feature extraction 
            and Flatten layers.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x