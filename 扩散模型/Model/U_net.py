import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

## 卷积模块（batch,dim,len)-->(batch,output_dim,len) (batch,output_dim,len/2)
class TCNBlock(nn.Module):
    def __init__(self,embedding_dim,output_dim,kernel_size = 3,dilation=1,drop=0.1):
        super(TCNBlock, self).__init__()
        self.tcn1 = nn.Conv1d(embedding_dim,output_dim, kernel_size, padding=(kernel_size - 2) * dilation, dilation=dilation)
        self.tcn2 = nn.Conv1d(output_dim, output_dim, kernel_size, padding=(kernel_size - 2) * dilation,dilation=dilation)
        self.norm1 = nn.BatchNorm1d(output_dim)
        self.norm2 = nn.BatchNorm1d(output_dim)
        self.silu = nn.SiLU()
        self.drop = nn.Dropout(drop)
        self.maxpool = nn.MaxPool1d(2)
    def forward(self, x):
        x1 = self.drop(self.silu(self.norm1(self.tcn1(x))))
        x2 = self.drop(self.silu(self.norm2(self.tcn2(x1))))
        out = self.maxpool(x2)
        return x2,out


# (batch,out_channels,len) --> (batch,out_channels,len)[(batch,out_channels,len/2)...]
class TCNUnetDown(nn.Module):
    def __init__(self, in_channels, out_channels,num_layers=2):
        super(TCNUnetDown, self).__init__()
        self.tcn_block = nn.ModuleList(
            [TCNBlock(embedding_dim = in_channels, output_dim=out_channels, kernel_size=3, drop=0.1) for _ in
             range(num_layers - 1)])
    def forward(self, x):
       maxsample = []
       for layer in self.tcn_block:
           x,sample = layer(x)
           maxsample.append(sample)
       return x,maxsample # 返回特征图和下采样后的特征图


#(batch,out_channels,len)[(batch,out_channels,len/2)...] -->(batch,out_channels,len)
class TCNUnetUp(nn.Module):
    def __init__(self, in_channels, out_channels,num_layers=2):
        super(TCNUnetUp, self).__init__()
        self.num_layers = num_layers
        self.upsample = nn.ModuleList([nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)for _ in
             range(num_layers - 1)])
        self.tcn_block = nn.ModuleList(
            [TCNBlock(embedding_dim=out_channels*2, output_dim=out_channels, kernel_size=3, drop=0.1) for _ in
             range(num_layers - 1)])
    def forward(self,x,sampel):
        for layer in range(self.num_layers-1):
            sampel[-(layer+1)] = self.upsample[layer](sampel[-(layer+1)])
            x = torch.cat((x, sampel[-(layer+1)]), dim=1)  # 拼接跳跃连接
            x,_ = self.tcn_block[layer](x)
        return x
# 输入 N 输出N*dim
def timestep_embedding(timesteps, dim,len, max_period=10000):
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding.unsqueeze(2).repeat(1, 1, len)

# class StepEmbed(nn.Module):
#     def __init__(self,step_dim,out_dim):
#         super(StepEmbed, self).__init__()
#         self.embed = nn.Sequential(
#             nn.Linear(step_dim, out_dim),
#             nn.SiLU(),
#             nn.Linear(out_dim, out_dim),
#         )
#     def forward(self,step):


#该模块的任务为将Encoder的中间产物加噪并去噪，最后训练出一个能够从纯噪声中生成正常中间产物的功能
# 输入：回望窗口嵌入、噪声、中间产物
# 输出：噪声
class diffusion_TCN(nn.Module):
    def __init__(self,in_channels, out_channels,deep=2):
        super(diffusion_TCN, self).__init__()
        ## 输入
        self.input_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)  # 输入卷积层
        ## 下采样
        self.u_down = TCNUnetDown(in_channels, out_channels,num_layers=deep)
        ## 上采样
        self.u_up =TCNUnetUp(out_channels,out_channels,num_layers=deep)
        ## 提示嵌入
        ## 输出
        self.out_conv = nn.Conv1d(out_channels, in_channels, kernel_size=1)  # 输出卷积层

    # (batch,dim,len)
    def forward(self,noise,step,prompt=None):
        input = self.input_conv(noise)
        if prompt != None:
            _, _, len_noise = noise.shape
            _,_,len_prompt = prompt.shape
            # 创建一个形状为 (batch, dim, len_2) 的零数组
            prompt = F.pad(prompt, (0, (len_noise-len_prompt)))
            input = input+prompt
        x,sample = self.u_down(input)
        _,dim,len = x.shape
        x = x+timestep_embedding(step,dim,len)
        x = self.u_up(x,sample)
        x = self.out_conv(x)
        return x

