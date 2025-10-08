class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, src):
        # src shape: [batch_size, hist_len, input_dim]
        outputs, (hidden, cell) = self.lstm(src)
        # hidden shape: [n_layers, batch_size, hidden_dim]
        # cell shape: [n_layers, batch_size, hidden_dim]
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, n_layers):
        super().__init__()
        self.output_dim = output_dim
        self.lstm = nn.LSTM(output_dim, hidden_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, cell):
        # input shape: [batch_size, 1, output_dim]
        # hidden, cell from encoder
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        # output shape: [batch_size, 1, hidden_dim]
        prediction = self.fc_out(output)
        # prediction shape: [batch_size, 1, output_dim]
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch_size, hist_len, input_dim]
        # trg: [batch_size, future_len, output_dim]
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        # 使用历史轨迹的最后一个点作为解码器的第一个输入
        input = src[:, -1, :].unsqueeze(1)

        for t in range(trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output.squeeze(1)

            # 决定是否使用 teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            # 如果是 teacher forcing，下一个输入是真实值；否则是当前预测值
            input = trg[:, t, :].unsqueeze(1) if teacher_force else output

        return outputs
