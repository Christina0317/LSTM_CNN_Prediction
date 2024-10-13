import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class CNNLSTMWithAttention(nn.Module):
    def __init__(self, model_params):
        super(CNNLSTMWithAttention, self).__init__()

        input_dim = model_params['input_dim']
        cnn_steps = model_params['cnn_steps']
        cnn_out_dim = model_params['cnn_out_dim']
        num_steps = model_params['num_steps']
        lstm_hidden_dim = model_params['lstm_hidden_dim']
        lstm_num_layers = model_params['lstm_num_layers']
        output_dim = model_params['output_dim']
        kernel_size = model_params['kernel_size']

        # CNN部分：输入相空间数据 (batch_size, time_steps, m)
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=cnn_out_dim, kernel_size=kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fn1 = nn.Linear(int((cnn_steps-2*(kernel_size-1))/2)*cnn_out_dim, num_steps)

        # LSTM部分：输入原始时间序列 + CNN 提取的特征 (因此input_size=2)
        self.lstm = nn.LSTM(input_size=num_steps, hidden_size=lstm_hidden_dim, num_layers=lstm_num_layers,
                            batch_first=True)

        # 注意力机制：计算注意力权重
        self.attention = nn.Linear(lstm_hidden_dim, 1)

        # 输出层
        self.fc = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, data):
        # CNN 处理相空间数据 (batch_size, time_steps, m)
        original_series, phase_space_data = data
        # original_series = original_series.float()
        # phase_space_data = phase_space_data.float()
        x = phase_space_data.transpose(1, 2)  # 调整为 (batch_size, m, time_steps)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        spatial_features = self.fn1(x)

        # 提取空间特征，形状为 (batch_size, time_steps, cnn_out_dim)
        # spatial_features = x.transpose(1, 2)  # 调整为 (batch_size, time_steps, cnn_out_dim)

        # 将空间特征和原始序列结合 -> shape(batch_size, 2, num_steps)
        combined_tensor = torch.stack((spatial_features, original_series), dim=1)

        # LSTM 输入结合后的特征
        lstm_out, _ = self.lstm(combined_tensor)

        # 注意力机制：计算每个时间步的注意力权重
        # lstm_out = lstm_out.transpose(1, 2)
        # shape(weights) = (batch_size, time_steps, 1)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)

        # 计算加权和：对时间步的 LSTM 输出进行加权
        weighted_lstm_out = torch.sum(lstm_out * attention_weights, dim=1)  # (batch_size, lstm_hidden_dim)

        # 输出层
        out = self.fc(weighted_lstm_out)
        return out[:, 0]