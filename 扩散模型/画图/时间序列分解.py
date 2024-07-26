# 定义模型类
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from tools.tools import StandardScaler


class ASinModel(nn.Module):
    def __init__(self, num_functions):
        super(ASinModel, self).__init__()
        # 初始化A, B, C, D为可学习的参数，每个参数是(num_functions, 1)的矩阵
        # self.A = nn.Parameter(torch.randn(num_functions, 1))
        # self.B = nn.Parameter(torch.randn(num_functions, 1))
        # self.C = nn.Parameter(torch.randn(num_functions, 1))
        # self.D = nn.Parameter(torch.randn(num_functions, 1))
        self.linear = nn.Linear(num_functions,num_functions,bias=True)
        self.linear_2 = nn.Linear(num_functions, num_functions, bias=True)
    def forward(self, x):
        # x 应该是 (batch_size, num_functions) 的形状
        y = torch.sin(self.linear(x))
        # 在第二个维度上进行求和
        y = self.linear_2(y)
        result = torch.sum(y, dim=1)
        return result


if __name__ == '__main__':
    # 获取数据
    path = '../../Informer2020/data/yeWei/yeWei.csv'
    data = pd.read_csv(path)["值"]
    mystn = StandardScaler()
    mystn.fit(data)
    data = mystn.transform(data)
    # 将循环序列扩展到与 DataFrame 行数相同的长度# 创建一个 1-80 循环的序列
    cycle_length = 80
    cycle = list(range(1, cycle_length + 1))
    repeated_cycle = (cycle * (len(data) // cycle_length + 1))[:len(data)]
    # 将 Series 和列表拼接成新的 Series
    combined_series = pd.concat([data, pd.Series(repeated_cycle)],axis=1, ignore_index=True)
    seg_train = int(len(combined_series) * 0.7)
    train_combined_series = torch.tensor(combined_series.loc[:seg_train].values, dtype=torch.float32)
    test_combined_series = torch.tensor(combined_series.loc[seg_train:].values, dtype=torch.float32)
    # 创建 TensorDataset
    train_dataset = TensorDataset(train_combined_series)
    test_dataset = TensorDataset(test_combined_series)

    # 创建 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    num_fun = 20
    getfunction = ASinModel(num_fun)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(getfunction.parameters(), lr=0.001)
    epochs = 20
    for epoch in range(epochs):
        getfunction.train()
        for item in train_dataloader:
            input = item[0][:,1].unsqueeze(1).expand(-1, num_fun)
            y =  item[0][:,0]
            out = getfunction(input)
            loss = criterion(y,out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    getfunction.eval()
    outs = []
    ys = []
    with torch.no_grad():
        for i, item in enumerate(test_dataloader):
            input = item[0][:, 1].unsqueeze(1).expand(-1, num_fun)
            y = item[0][:, 0]
            out = getfunction(input)
            out = out.numpy()
            y = y.numpy()
            outs = np.concatenate((outs, out))
            ys = np.concatenate((ys, y))

    # 绘制第n步结果对比
    plt.plot(range(len(ys)), outs)
    plt.plot(range(len(ys)), ys)
    # 添加图例
    plt.legend()
    # 显示图表
    plt.show()