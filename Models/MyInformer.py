import math

import torch
import torch.nn as nn
import torch.optim as optim


class AattentionLayer(nn.Module):
    def __init__(self,input_dim,num_heads,head_dim,output_dim):
        super(AattentionLayer, self).__init__()
        self.dim = num_heads*head_dim
        self.head =num_heads
        self.head_dim =head_dim
        self.qkv_layer = nn.Linear(input_dim,self.dim*3)
        self.fc_out = nn.Linear(self.dim, output_dim)
        self.soft = nn.Softmax(dim=3)
    def forward(self,data,mask=None):
        N = data.shape[0]
        L = data.shape[1]
        qkv = self.qkv_layer(data)
        q = qkv[:,:,:self.dim].reshape(N,L,self.head,self.head_dim)
        k = qkv[:,:,self.dim:self.dim*2].reshape(N,L,self.head,self.head_dim)
        v = qkv[:,:,self.dim*2:].reshape(N,L,self.head,self.head_dim)

        scores = torch.einsum("nqhd,nkhd->nhqk",[q,k])
        if mask is not None:
            mask = mask.unsqueeze(1)
            # 使用 expand 将矩阵扩展为 (128, 2, 120, 120)
            mask = mask.expand(-1, scores.shape[1], -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = self.soft(scores/(self.head_dim**(1/2)))
        out = torch.einsum("nhql,nlhd->nqhd",[attention,v]).reshape(N,L,self.dim)
        out = self.fc_out(out)
        return out

class crossattentionLayer(nn.Module):
    def __init__(self,input_dim,num_heads,head_dim,return_dim):
        super(crossattentionLayer, self).__init__()
        self.dim = num_heads*head_dim
        self.head =num_heads
        self.head_dim =head_dim
        self.kv_layer = nn.Linear(input_dim,self.dim*2)
        self.q_layer = nn.Linear(input_dim,self.dim)
        self.fc_out = nn.Linear(self.dim, return_dim)
        self.soft = nn.Softmax(dim=3)

    def forward(self, data, input, mask = None):
        N = data.shape[0]
        # L = data.shape[1]
        kv = self.kv_layer(data)
        k = kv[:,:,:self.dim].reshape(N,-1,self.head,self.head_dim).transpose(1, 2)
        v = kv[:,:,self.dim:].reshape(N,-1,self.head,self.head_dim).transpose(1, 2)
        q = self.q_layer(input).reshape(N,-1,self.head,self.head_dim).transpose(1, 2)

        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v).transpose(1, 2).reshape(N,-1,self.dim)

        output = self.fc_out(output)
        return output,attn


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


class Decoder(nn.Module):
    def __init__(self,input_dim,num_heads,head_dim,attention_dim,ff_dim,dropout=0.1):
        super(Decoder, self).__init__()
        self.attentionlayer = AattentionLayer(input_dim=input_dim,num_heads=num_heads,head_dim=head_dim,output_dim=attention_dim)
        self.crossattentionlayer =crossattentionLayer(input_dim=attention_dim,num_heads=num_heads,head_dim=head_dim,return_dim=ff_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self,data,input,mask):
        attention_out = self.attentionlayer(input,mask)
        x = self.dropout(attention_out)
        x = self.crossattentionlayer(data,x)
        return x




class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

class Output(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(Output, self).__init__()
        self.fc_1 = nn.Linear(input_dim,hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x

class myInformer(nn.Module):
    def __init__(self,input_dim,embed_dim,num_heads,head_dim,attention_dim,ff_dim,num_layers,de_num_layers):
        super(myInformer, self).__init__()

        self.enc_embedding = DataEmbedding(c_in = input_dim, d_model=embed_dim)
        self.dec_embedding = DataEmbedding(c_in = input_dim, d_model=embed_dim)

        self.encoders = nn.ModuleList(
            [Encoder(input_dim= embed_dim, num_heads = num_heads, head_dim = head_dim, attention_dim = attention_dim, ff_dim = ff_dim, dropout=0.1) for _ in
             range(num_layers - 1)])

        self.decoders = nn.ModuleList(
            [Decoder(input_dim= embed_dim, num_heads = num_heads, head_dim = head_dim, attention_dim = embed_dim, ff_dim = input_dim, dropout=0.1) for _ in
             range(de_num_layers - 1)])

        self.outpu = Output(input_dim=attention_dim,hidden_dim=head_dim,output_dim=input_dim)

    def forward(self,input,decoder_input):
        # 将input切分为encoder_input和decoder_input
        encoder_input = input
        decoder_input = decoder_input

        encode = self.enc_embedding(encoder_input)
        decode = self.dec_embedding(decoder_input)
        B,L,D = decode.shape
        # 创建一个形状为 (seq_len, seq_len) 的矩阵，其中下三角部分为1，上三角部分为0
        mask = torch.tril(torch.ones((L, L))).unsqueeze(0)
        # 将矩阵扩展为 (batch_size, seq_len, seq_len)
        mask = mask.expand(B, -1, -1).to('cuda:0')
        for layer in self.encoders:
            encode = layer(encode)
        for layer in range(len(self.decoders)):
            if layer ==0 :
                out,_ = self.decoders[layer](encode,decode,mask)
            else:
                out,_ = self.decoders[layer](encode,out,mask)
        # out =self.outpu(out)
        return out

