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


class OutputLayer(nn.Module):
    def __init__(self,input_dim,output_dim,input_len,output_len):
        super(OutputLayer, self).__init__()
        self.fc_1 = nn.Linear(input_dim,output_dim)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(input_len,output_len)
    def forward(self,input):
        x = self.relu(self.fc_1(input))
        x = x.transpose(1,2)
        out = self.fc_2(x).transpose(1,2)
        return out
## 写一个Transform用来进行时间序列预测，encoder使用自注意力机制，输出使用其他方式
class TimeTransform(nn.Module):
    def __init__(self,input_dim,num_heads,head_dim,attention_dim,ff_dim,input_len,output_len,dropout=0.1):
         super(TimeTransform, self).__init__()
         self.encoder = Encoder(input_dim,num_heads,head_dim,attention_dim,ff_dim,dropout=0.1)
         self.output = OutputLayer(input_dim = attention_dim,output_dim = input_dim,input_len = input_len,output_len = output_len)

    def forward(self,data):
        out = self.encoder(data)
        out = self.output(out)
        return out