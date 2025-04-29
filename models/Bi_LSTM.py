import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from torch.autograd import Variable
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BiLSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(BiLSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size*2, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)).to(device)
        c_0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)).to(device)
        
        out, (hn, cn) = self.bilstm(x, (h_0, c_0))
        out = self.dropout(out)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
