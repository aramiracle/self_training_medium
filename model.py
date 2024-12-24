import torch.nn as nn

def create_model():
    """
    Creates and returns a custom Convolutional Neural Network (CNN) model for image classification.

    The model consists of three convolutional layers, each followed by batch normalization, ReLU activation, 
    max pooling, and dropout. The output of the convolutional layers is passed through two fully connected layers
    to produce the final classification output.

    The architecture is as follows:
        - 3 Convolutional layers with increasing filter sizes (8, 16, 32).
        - Max pooling and dropout after each convolutional layer.
        - Flatten the output from the convolutional layers before passing to the fully connected layers.
        - Fully connected layers: one hidden layer with 128 units and dropout, followed by an output layer with 10 units (for 10 classes).
    
    Returns:
        nn.Module: A PyTorch model instance of the CustomCNN class.

    """
    def single_conv_layer(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        )

    class CustomCNN(nn.Module):
        def __init__(self):
            super(CustomCNN, self).__init__()
            self.conv_layers = nn.Sequential(
                single_conv_layer(1, 8),
                single_conv_layer(8, 16),
                single_conv_layer(16, 32)
            )

            self.fc_layers = nn.Sequential(
                nn.Linear(32 * 3 * 3, 128),  # Adjust based on the input image size.
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 10)
            )

        def forward(self, x):
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layers.
            x = self.fc_layers(x)
            return x
            
    return CustomCNN()
