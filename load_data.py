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
import random


from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#load daily gesture data
class HCPDDataset(Dataset):
    def __init__(self, xlsx_file, transform=None):
        self.total_data = pd.read_excel(xlsx_file)
        self.data = self.total_data.iloc[:,1:].values
        # self.data = self.total_data.iloc[0:114,4:].values

        # scaler = MinMaxScaler(feature_range=(0, 1))
        # self.data = scaler.fit_transform(self.data)

        # print(self.data[0])
        # print(type(self.data))


        self.label = self.total_data.iloc[:,0].values
        # self.label = self.total_data.iloc[:,1].values
        # print(self.label)

        self.label[self.label=='HC']=0
        self.label[self.label=='PD']=1
        # self.transform = transform

        # print(self.label[0])
        # print(type(self.label))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        # y = torch.tensor(np.array(self.label[idx]).astype(float), dtype=torch.float32)
        y = torch.tensor(self.label[idx], dtype=torch.float32)

        x = x.to(device)
        y = y.to(device)

        # if self.transform:
        #     x = self.transform(x)

        return x,y


class TimeShift:
    def __init__(self, max_shift=10):
        self.max_shift = max_shift

    def __call__(self, x):
        shift = np.random.randint(-self.max_shift, self.max_shift)
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy() # move tensor to CPU and convert to numpy array
        x = np.roll(x, shift)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(device=torch.device('cuda')) # convert back to tensor and move to GPU
        return x

class AmplitudeScale(object):
    def __init__(self, max_scale):
        self.max_scale = max_scale

    def __call__(self, x):
        scale = np.random.uniform(1.0, self.max_scale)
        return x * scale

class Jitter(object):
    def __init__(self, max_jitter=0.1):
        self.max_jitter = max_jitter

    def __call__(self, x):
        noise = torch.randn_like(x) * self.max_jitter
        return x + noise

class Scaling:
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, sample):
        x= sample
        scaled_x = x * self.scale_factor
        return scaled_x
class TimeWarp:
    def __call__(self, sample):
        x = sample
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy() # move tensor to CPU and convert to numpy array
        x_flat = x.flatten()  # Flatten the input array
        t = np.arange(len(x_flat))
        warp_func = np.random.uniform(-1, 1)
        warped_t = t + warp_func * np.sin(2 * np.pi * t / len(x_flat))
        warped_x = np.interp(t, warped_t, x_flat)
        return warped_x.reshape(x.shape)  # Reshape back to the original shape



import torch

class FourierAugmentation:
    def __init__(self, max_phase_shift, max_frequency_shift):
        self.max_phase_shift = max_phase_shift
        self.max_frequency_shift = max_frequency_shift

    def __call__(self, sample):
        x = sample
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()  # move tensor to CPU and convert to numpy array

        n = len(x)
        t = np.arange(n)
        frequencies = np.fft.fftfreq(n)
        amplitudes = np.abs(np.fft.fft(x, axis=0))

        phase_shifts = np.random.uniform(-self.max_phase_shift, self.max_phase_shift, size=(64, 64))
        frequency_shifts = np.random.uniform(-self.max_frequency_shift, self.max_frequency_shift, size=(64, 64))

        phase_shifted_x = amplitudes * np.exp(1j * (np.angle(np.fft.fft(x, axis=0)) + phase_shifts))
        frequency_shifts = frequency_shifts.reshape(-1, 1)  # Reshape to match frequencies shape
        frequency_shifted_x = np.real(
            np.fft.ifft(np.fft.fft(phase_shifted_x, axis=0) * np.exp(2j * np.pi * frequencies * frequency_shifts)))

        return frequency_shifted_x

class WindowSlicing:
    def __init__(self, window_size):
        self.window_size = window_size

    def __call__(self, sample):
        x = sample
        n = len(x)
        if n <= self.window_size:
            raise ValueError("Input sample length should be larger than window size.")
        start = np.random.randint(0, n - self.window_size + 1)
        end = start + self.window_size
        sliced_x = x[start:end]
        return sliced_x

# dataset = HCPDDataset('dataset/DATA_HC_PD_v1.0_final_jsk_removed_features_normalized.xlsx')
# dataset = HCPDDataset('dataset/DATA_HC_PD_v1.0_final_jsk_removed_features_mean_0_std_1.xlsx')
# dataset = HCPDDataset('dataset/DATA_HC_PD_lowerlimbs.xlsx')
# dataset = HCPDDataset('dataset/DATA_HC_PD_upperlimbs.xlsx')
dataset = HCPDDataset('dataset/DATA_HC_PD_total.xlsx')



print(dataset[1][0].shape)
all_shapes = [data[0].shape for data in dataset]
total_shape = (len(dataset),) + all_shapes[0][1:]

print(total_shape)



