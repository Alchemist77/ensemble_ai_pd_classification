import torch
import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self, input_size, num_classes,input_data_size):
        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Calculate the input size for the fully connected layer
        with torch.no_grad():
            x = torch.zeros(1, 1, input_data_size)
            x = self.conv1(x)
            x = nn.ReLU()(x)
            x = nn.MaxPool1d(kernel_size=2)(x)
            x = self.conv2(x)
            x = nn.ReLU()(x)
            x = nn.MaxPool1d(kernel_size=2)(x)
            self.fc_input_size = x.view(1, -1).size(1)

        self.fc = nn.Linear(self.fc_input_size, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool1d(kernel_size=2)(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool1d(kernel_size=2)(x)
        x = nn.Flatten()(x)
        x = self.fc(x)

        return x
