import torch
import torch.nn as nn


class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_size=2, num_layers=2, pred_len=12):
        """
        一个简单的LSTM模型，用于轨迹预测。
        :param input_size: 每个时间步的输入特征维度 (例如: x, y -> 2)
        :param hidden_size: LSTM隐藏层维度
        :param output_size: 每个时间步的输出特征维度 (例如: x, y -> 2)
        :param num_layers: LSTM层数
        :param pred_len: 预测未来轨迹的长度
        """
        super(TrajectoryLSTM, self).__init__()
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层，处理序列输入
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 全连接层，将LSTM的输出映射到最终的预测轨迹
        self.fc = nn.Linear(hidden_size, pred_len * output_size)

    def forward(self, x):
        # x 的形状: (batch_size, seq_len, input_size)

        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))

        # 我们只关心序列的最后一个时间步的输出
        # out 的形状: (batch_size, seq_len, hidden_size)
        # last_output 的形状: (batch_size, hidden_size)
        last_output = out[:, -1, :]

        # 通过全连接层进行预测
        # pred 的形状: (batch_size, pred_len * output_size)
        pred = self.fc(last_output)

        # 重塑输出以匹配 (batch_size, pred_len, output_size)
        pred = pred.view(x.size(0), self.pred_len, -1)

        return pred

