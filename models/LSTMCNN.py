import torch
import torch.nn as nn
from load_data import HCPDDataset, TimeShift, AmplitudeScale, Jitter,Scaling,TimeWarp,FourierAugmentation,WindowSlicing

class LSTMCNN(nn.Module):
    def __init__(self, input_size):
        super(LSTMCNN, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=1, dropout=0.5, batch_first=True)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(960, 512) #(32, 1, 190),#(32, 1, 78),#(32, 1, 112)
        self.fc2 = nn.Linear(512, 2)
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, x):
        # LSTM layers
        # print("x1",x.shape)
        x, _ = self.lstm(x)
        # print("x2",x.shape)

        
        # Convolutional layers
        # x = x.permute(0, 2, 1)  # Reshape for convolutional layer
        # print("x3",x.shape)
        x = self.conv1(x)
        # print("x4",x.shape)
        x = self.maxpool1(x)
        # print("x5",x.shape)
        x = self.conv2(x)
        # print("x6",x.shape)
        x = self.maxpool2(x)
        # print("x7",x.shape)
        
        # Flatten
        x = x.view(x.size(0), -1)
        # print("x8",x.shape)
        
        # Dense layers
        x = self.dropout(x)
        # print("x9",x.shape)
        x = self.fc1(x)
        # print("x10",x.shape)
        x = self.fc2(x)
        # print("x11",x.shape)

        # Apply sigmoid activation
        x = self.sigmoid(x)
        # print("x12",x.shape)
        
        return x

# # Test the forward pass
# dataset = HCPDDataset('dataset/DATA_HC_PD_lowerlimbs.xlsx')  # using with normalization 0 to 1
# input_size = dataset[0][0].shape[-1]
# model = StackedLSTMConvNet(input_size)
# x = torch.randn(32, 1, 78)
# output = model(x)
# print(output.shape)
