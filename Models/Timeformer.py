import torch
import torch.nn as nn

class AattentionLayer(nn.Module):
    def __init__(self,input_dim,num_heads,head_dim,output_dim):
        super(AattentionLayer, self).__init__()
        self.dim = num_heads*head_dim
        self.head =num_heads
        self.head_dim =head_dim
        self.qkv_layer = nn.Linear(input_dim,self.dim*3)
        self.fc_out = nn.Linear(self.dim, output_dim)
        self.soft = nn.Softmax(dim=3)
    def forward(self,data):
        N = data.shape[0]
        L = data.shape[1]
        qkv = self.qkv_layer(data)
        q = qkv[:,:,:self.dim].reshape(N,L,self.head,self.head_dim)
        k = qkv[:,:,self.dim:self.dim*2].reshape(N,L,self.head,self.head_dim)
        v = qkv[:,:,self.dim*2:].reshape(N,L,self.head,self.head_dim)

        scores = torch.einsum("nqhd,nkhd->nhqk",[q,k])
        attention = self.soft(scores/(self.head_dim**(1/2)))
        out = torch.einsum("nhql,nlhd->nqhd",[attention,v]).reshape(N,L,self.dim)
        out = self.fc_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(FeedForward, self).__init__()
        self.fc_1 = nn.Linear(input_dim,hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim,output_dim)
        self.relu = nn.ReLU()
    def forward(self,input):
        output = self.relu(self.fc_1(input))
        out = self.fc_2(output)
        return out


class Encoder(nn.Module):
    def __init__(self,input_dim,num_heads,head_dim,attention_dim,ff_dim,dropout=0.1):
        super(Encoder, self).__init__()
        self.attentionlayer = AattentionLayer(input_dim=input_dim,num_heads=num_heads,head_dim=head_dim,output_dim=attention_dim)
        self.fc = FeedForward(input_dim =attention_dim,hidden_dim=ff_dim,output_dim =attention_dim)
        self.norm1 = nn.LayerNorm(attention_dim)
        self.norm2 = nn.LayerNorm(attention_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self,input):
        attention_out = self.attentionlayer(input)
        x = self.dropout(self.norm1(attention_out+input))
        ff_out = self.fc(x)
        out = self.dropout(self.norm2(x+ff_out))
        return out

## 对调fc_1和fc_2,从实验来看模型预测能力很大程度上依赖于最后一层的输出能力
## 设计一个好的输出层比实际好的学习层更加重要,但是过于复杂的输出层会让模型收敛变得缓慢
class OutputLayer(nn.Module):
    def __init__(self,input_dim,output_dim,input_len,output_len):
        super(OutputLayer, self).__init__()
        self.fc_1 = nn.Linear(input_dim,output_dim)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(input_len,output_len)
    def forward(self,input):
        x = self.relu(self.fc_1(input))
        x = x.transpose(1, 2)
        out = self.fc_2(x).transpose(1, 2)
        return out


## 写一个Transform用来进行时间序列预测，encoder使用自注意力机制，输出使用其他方式
class Timeformer(nn.Module):
    def __init__(self,input_dim,num_heads,head_dim,attention_dim,ff_dim,input_len,output_len,num_layers = 6,dropout=0.1):
         super(Timeformer, self).__init__()
         self.num_layers = num_layers
         self.encoders = nn.ModuleList([Encoder(input_dim,num_heads,head_dim,attention_dim,ff_dim,dropout=0.1) for _ in range(num_layers - 1)])
         self.encoder = Encoder(input_dim,num_heads,head_dim,attention_dim,ff_dim,dropout=0.1)
         self.output = OutputLayer(input_dim = attention_dim,output_dim = input_dim,input_len = input_len,output_len = output_len)

    def forward(self,data):
        for layer in  self.encoders:
            out = layer(data)
        out = self.output(out)
        return out