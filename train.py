import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch import nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import Subset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

class StreamToLogger(object):
    """
    自定义流，用于同时将输出信息发送到标准输出和文件。
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # 这个函数在这里是为了兼容文件对象的接口
        self.terminal.flush()
        self.log.flush()

sys.stdout = StreamToLogger("console_output_TCN.txt")

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = self.network(x)
        return self.linear(x[:, :, -1])

class TCNModel(nn.Module):
    def __init__(self, input_dim, output_dim, seq_out_len, num_channels, kernel_size, dropout):
        super(TCNModel, self).__init__()
        self.tcn = TCN(input_dim, output_dim * seq_out_len, num_channels, kernel_size, dropout)
        self.output_dim = output_dim

    def forward(self, x):
        out = self.tcn(x.transpose(1, 2))
        return out.view(x.size(0), -1, self.output_dim)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

data = pd.read_excel('17+18-已处理.xlsx')
data = data.iloc[1:, 1:]

look_back = 4

# 窗口函数
# 修改split_sequence函数来生成多步标签
def split_sequence(sequence, look_back, seq_out_len):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + look_back
        out_end_ix = end_ix + seq_out_len
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence.iloc[i:end_ix, :].values, sequence.iloc[end_ix:out_end_ix, :].values
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)




# 初始数据读取和处理
x, y = split_sequence(data, look_back=look_back, seq_out_len=1)  # 初始值仅用于获取数据维度
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 设置循环
seq_out_lens = [1, 4, 8, 13, 16]

for seq_out_len in seq_out_lens:
    print('seq_out_len:', seq_out_len)
    x, y = split_sequence(data, look_back=look_back, seq_out_len=seq_out_len)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # 数据转换
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    # DataLoader
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # 初始化指标存储列表
    mse_scores, rmse_scores, mae_scores, r2_scores, mape_scores = [], [], [], [], []

    # 运行模型 10 次
    epoches = 50
    num_runs = 10
    num_channels = [25, 50, 100]  # 示例通道数
    kernel_size = 3  # 示例核心大小
    dropout = 0.2  # dropout率
    for run in range(num_runs):
        # 模型定义
        model = TCNModel(input_dim=X_train.shape[-1], output_dim=14, seq_out_len=seq_out_len, num_channels=num_channels,
                         kernel_size=kernel_size, dropout=dropout)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # 模型训练
        model.to(device)
        for epoch in range(epoches):
            model.train()
            train_loss = 0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

        # 模型测试
        model.eval()
        preds = model(X_test.to(device))
        predictions = preds.view(-1, seq_out_len, y_train.shape[-1]).cpu().detach().numpy()
        predictions_reshaped = predictions.reshape(-1, y_train.shape[-1])
        y_test_reshaped = y_test.numpy().reshape(-1, y_train.shape[-1])

        # 计算评估指标
        mse = mean_squared_error(y_test_reshaped, predictions_reshaped)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_reshaped, predictions_reshaped)
        r2 = r2_score(y_test_reshaped, predictions_reshaped)
        mape = np.mean(np.abs((y_test_reshaped - predictions_reshaped) / y_test_reshaped)) * 100

        # 存储指标
        mse_scores.append(mse)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        mape_scores.append(mape)

        # 打印当前迭代的评估指标
        print(f'Run {run + 1}:')
        print('MSE:', mse)
        print('RMSE:', rmse)
        print('MAE:', mae)
        print('R-squared:', r2)
        print('MAPE:', mape)
        print('-' * 50)

    # 计算平均评估指标
    avg_mse = np.mean(mse_scores)
    avg_rmse = np.mean(rmse_scores)
    avg_mae = np.mean(mae_scores)
    avg_r2 = np.mean(r2_scores)
    avg_mape = np.mean(mape_scores)


    # 打印平均评估指标
    print('Average Metrics Over 10 Runs for seq_out_len =', seq_out_len)
    print('Average MSE:', avg_mse)
    print('Average RMSE:', avg_rmse)
    print('Average MAE:', avg_mae)
    print('Average R-squared:', avg_r2)
    print('Average MAPE:', avg_mape)
    print('-' * 100)