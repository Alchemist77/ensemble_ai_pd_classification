import torch
import torch.nn as nn

class CNNBGRU(nn.Module):
    def __init__(self):
        super(CNNBGRU, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.maxpool = nn.MaxPool1d(kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=48, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=48, out_channels=32, kernel_size=3)
        # self.dense1 = nn.Linear(1856, 256)  #(32, 1, 190) 
        #self.dense1 = nn.Linear(672, 256)  # (32, 1, 78)
        self.dense1 = nn.Linear(1024, 256)  # (32, 1, 112)

        self.dropout1 = nn.Dropout(0.2)
        self.bgru = nn.GRU(input_size=256, hidden_size=32, bidirectional=True)
        self.dense2 = nn.Linear(64, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.dense3 = nn.Linear(256, 2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # print("x1", x.shape)
        x = self.conv1(x)
        # print("x2", x.shape)
        x = self.maxpool(x)
        # print("x3", x.shape)
        x = self.conv2(x)
        # print("x4", x.shape)
        x = self.conv3(x)
        # print("x5", x.shape)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        # print("x6", x.shape)
        x = self.dense1(x)
        # print("x7", x.shape)
        x = self.dropout1(x)
        # print("x8", x.shape)
        x, _ = self.bgru(x.unsqueeze(0))
        # print("x9", x.shape)
        x = x.squeeze(0)
        # print("x10", x.shape)
        x = self.dense2(x)
        # print("x11", x.shape)
        x = self.dropout2(x)
        # print("x12", x.shape)
        x = self.dense3(x)
        # print("x13", x.shape)
        x = self.sigmoid(x)
        # print("x14", x.shape)
        return x

# Create an instance of the model
model = CNNBGRU()
x = torch.randn(32, 1, 112)
output = model(x)
print(output.shape)