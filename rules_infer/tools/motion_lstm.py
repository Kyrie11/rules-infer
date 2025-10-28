import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import random

class Encoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
    def forward(self, src):
        outputs, (hidden, cell) = self.lstm(src)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim=2, hidden_dim=32, n_layers=2):
        super().__init__()
        self.output_dim = output_dim
        self.lstm = nn.LSTM(output_dim, hidden_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    def forward(self, input, hidden, cell):
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        prediction = self.fc_out(output)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    def forward(self, src, future_len, teacher_forcing_ratio=0.0):
        batch_size = src.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, future_len, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        input = src[:, -1, :].unsqueeze(1)
        for t in range(future_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output.squeeze(1)
            input = output
        return outputs
