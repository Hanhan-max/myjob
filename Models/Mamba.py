import torch.nn as nn

import torch.nn as nn
import torch
from einops import rearrange, repeat, einsum
from torch.nn.utils import weight_norm

class SSM(nn.Module):
    def __init__(self,N,D,L):
        super(SSM, self).__init__()
        self.L = L
        self.d = D
        self.N = N
        # 创建A矩阵
        A = repeat(torch.arange(1, N + 1), 'n -> d n', d=D)
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(D))
        self.s_b_c = nn.Linear(D,N*2)
        self.s_1 = nn.Linear(D,1)
        self.s_delta = nn.Linear(1,D)
        self.softplus = nn.Softplus()
    def forward(self,x):
        '''

        :param x: (B,L,D)
        :return: y(B,L,D)
        '''

        (b, l, d) = x.shape
        n = self.N

        A = -torch.exp(self.A_log.float())  # shape (D,N)
        D = self.D.float()
        BandC = self.s_b_c(x)
        B,C = BandC[:,:,:self.N],BandC[:,:,self.N:]
        delta = self.softplus(self.s_delta(self.s_1(x)))

        # 离散化A和B 见论文公式（4）
        deltaA = torch.exp(einsum(delta, A, 'b l d, d n -> b l d n'))
        deltaB_u = einsum(delta, B, x, 'b l d, b l n, b l d  -> b l d n')

        h = torch.zeros((b, d, n), device=deltaA.device)
        ys = []
        for i in range(l):
            h = deltaA[:,i] * h + deltaB_u[:,i]
            y = einsum(h, C[:, i, :], 'b d n, b n -> b d')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d)

        y = y

        return y


class Mamba_block(nn.Module):
    def __init__(self, L, n_layers, D, N, ):
        super(Mamba_block, self).__init__()
        self.L = L
        self.n_layers = n_layers
        self.D = D
        self.N = N
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.linear_x1 = nn.Linear(1, D*2)
        # self.linear_x2 = nn.Linear(1, D)
        # [B, L, D]
        self.conv = nn.Conv1d(D, D, 1)
        self.ssm = SSM(N,D,L)
        self.linear_x3 = nn.Linear(D, 1)
    def forward(self,input):
        # x =input
        # res_x = input
        # [16,256,1]
        x1_out = self.linear_x1(input)
        x,res_x = x1_out[:,:,:self.D],x1_out[:,:,self.D:]
        res_x = self.silu(res_x)

        x = x.transpose(2,1)
        x=self.conv(x)
        x = x.transpose(2, 1)
        x = self.silu(x)
        x = self.ssm(x)

        # 使用 torch.mul() 函数进行同位置相乘
        out = torch.mul(x, res_x)

        y = self.linear_x3(out)
        return y



class Mamba(nn.Module):
    # L, n_layers, D, N
    def __init__(self,L,D,N, laryer_num,out_len, output_size=1):
        super(Mamba, self).__init__()
        layers = []

        for i in range(laryer_num):
            layers += [Mamba_block( L, laryer_num, D, N)]

        self.network = nn.Sequential(*layers)
        self.out = nn.Linear(L,out_len)
    def forward(self, x):
        x = self.network(x)
        x = x.transpose(1,2)
        x = self.out(x).transpose(1,2)
        return x
