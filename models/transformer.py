import torch
import torch.nn as nn
from torch.nn import Transformer

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))

        # Compute the positional encodings
        encodings = torch.zeros(max_len, d_model)
        encodings[:, 0::2] = torch.sin(position * div_term)
        encodings[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('encodings', encodings.unsqueeze(0))

    def forward(self, x):
        x = x + self.encodings[:, :x.size(1)]
        return x

# Define the Transformer model for time series classification
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_heads, num_layers):
        super(TimeSeriesTransformer, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # print("x",x.shape)
        x = x.permute(0, 2, 1)
        # print("x",x.shape)
        x = self.embedding(x)
        x = self.positional_encoding(x)

        x = self.transformer_encoder(x)

        # Global average pooling
        x = torch.mean(x, dim=1)

        x = self.fc(x)
        return x
