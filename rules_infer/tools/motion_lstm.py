import torch
import torch.nn as nn


class TrajectoryLSTM(nn.Module):
    def __init__(self, config):
        super(TrajectoryLSTM, self).__init__()
        self.hist_len = config.HIST_LEN
        self.pred_len = config.PRED_LEN
        self.input_dim = config.INPUT_DIM
        self.output_dim = config.OUTPUT_DIM

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(128, self.pred_len * self.output_dim)

    def forward(self, x):
        # x shape: (batch_size, hist_len, input_dim)
        _, (hidden, _) = self.lstm(x)
        # hidden shape: (num_layers, batch_size, hidden_size)
        # 我们只使用最后一层的最后一个时间步的隐藏状态
        last_layer_hidden = hidden[-1]  # shape: (batch_size, hidden_size)

        output = self.fc(last_layer_hidden)
        # output shape: (batch_size, pred_len * output_dim)

        # Reshape to (batch_size, pred_len, output_dim)
        output = output.view(-1, self.pred_len, self.output_dim)
        return output

