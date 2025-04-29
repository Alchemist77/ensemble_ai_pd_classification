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

from load_data import HCPDDataset, TimeShift, AmplitudeScale, Jitter
from sklearn.model_selection import train_test_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EnsembleModel(nn.Module):
    def __init__(self, lstm_model, gru_model, bilstm_model, cnn_model, bigru_model, num_classes, input_size):
        super(EnsembleModel, self).__init__()
        self.lstm_model = lstm_model.to(device)
        self.gru_model = gru_model.to(device)
        self.bilstm_model = bilstm_model.to(device)
        self.cnn_model = cnn_model.to(device)
        self.bigru_model = bigru_model.to(device)
        self.num_classes = num_classes
        
        # Calculate the total output size from individual models
        total_output_size = self._get_total_output_size(input_size)
        
        self.classifier = nn.Linear(total_output_size, num_classes)
        
    def forward(self, x):
        lstm_out = self.lstm_model(x)
        gru_out = self.gru_model(x)
        bilstm_out = self.bilstm_model(x)
        cnn_out = self.cnn_model(x)
        bigru_out = self.bigru_model(x)
        
        # Concatenate the outputs from individual models along the last dimension
        out = torch.cat((lstm_out, gru_out, bilstm_out, cnn_out, bigru_out), dim=-1)
        
        out = self.classifier(out)
        return out
    
    def _get_total_output_size(self, input_size):
        # Create a dummy tensor and pass it through the individual models
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_size)
            dummy_input = dummy_input.to(device)
            lstm_out = self.lstm_model(dummy_input)
            gru_out = self.gru_model(dummy_input)
            bilstm_out = self.bilstm_model(dummy_input)
            cnn_out = self.cnn_model(dummy_input)
            bigru_out = self.bigru_model(dummy_input)
        
        # Sum up the output sizes from individual models
        total_output_size = lstm_out.size(-1) + gru_out.size(-1) + bilstm_out.size(-1) + cnn_out.size(-1) + bigru_out.size(-1)
        
        return total_output_size