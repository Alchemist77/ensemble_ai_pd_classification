import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
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


# Define the BiGRU model
class BiGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiGRUModel, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :]) # Use only the last output of the sequence for classification
        out = torch.sigmoid(out) # Sigmoid activation for binary classification
        return out
