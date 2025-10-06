# model.py
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers):
        super(Decoder, self).__init__()
        # Decoder的输入维度是2 (x, y)，而不是完整的特征维度
        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, config):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(config.INPUT_DIM, config.HIDDEN_DIM, config.NUM_LAYERS)
        self.decoder = Decoder(config.OUTPUT_DIM, config.HIDDEN_DIM, config.NUM_LAYERS)
        self.pred_len = config.PRED_LEN
        self.device = config.DEVICE

    def forward(self, src, target):
        # src shape: (batch_size, obs_len, input_dim)
        # target shape: (batch_size, pred_len, output_dim)

        batch_size = src.shape[0]
        outputs = torch.zeros(batch_size, self.pred_len, self.decoder.fc.out_features).to(self.device)

        hidden, cell = self.encoder(src)

        # Decoder的第一个输入是观察序列的最后一个点的(x, y)坐标
        # 在我们的数据预处理中，这个点是相对坐标 (0,0)
        decoder_input = src[:, -1, :2].unsqueeze(1)  # 只取前两位(x, y)

        # Teacher Forcing: 使用真实的target作为下一步的输入
        for t in range(self.pred_len):
            decoder_output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t, :] = decoder_output.squeeze(1)
            decoder_input = target[:, t, :].unsqueeze(1)

        return outputs
