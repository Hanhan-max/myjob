import torch
import torch.nn as nn
# 自定义损失函数类
class MyMseLoss(nn.Module):
    def __init__(self):
        super(MyMseLoss, self).__init__()

    def forward(self, output, target):
        # 计算损失
        loss = torch.mean((output - target) ** 2)  # 简单的均方误差
        return loss

class weightLoss(nn.Module):
    def __init__(self):
        super(weightLoss, self).__init__()

    def forward(self, output, target):
        b,l=output.shape[0],output.shape[1]
        w = torch.arange(l, 0, -1).unsqueeze(0).unsqueeze(2).repeat(128, 1, 1)
        w =(w/sum(range(l))).to('cuda:0')
        # 计算损失
        loss = torch.mean((output - target) ** 2)  # 简单的均方误差
        return loss
