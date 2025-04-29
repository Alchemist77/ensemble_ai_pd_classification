import torch
import torch.nn as nn

class PDCNN(nn.Module):
    def __init__(self):
        super(PDCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        # self.dense1 = nn.Linear(128 * 91, 512) #(32, 1, 190)
        # self.dense1 = nn.Linear(4480, 512) #(32, 1, 78)
        self.dense1 = nn.Linear(6656, 512) #(32, 1, 112)

        self.dense2 = nn.Linear(512, 256) 
        self.dense3 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 2)  # Two-class classification output

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.output(x)
        return x

# Test the forward pass
model = PDCNN()
x = torch.randn(32, 1, 112)
output = model(x)
print(output.shape)
