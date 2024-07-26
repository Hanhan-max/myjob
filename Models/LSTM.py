import torch
import torch.nn as nn

class myLSTM(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_layer):
        super(myLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,hidden_size=input_dim,num_layers=num_layer, batch_first=True)
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.num_layers = num_layer
        self.out = nn.Linear(480,120)
    def forward(self,data):
        h0 = torch.zeros(self.num_layers, data.size(0), self.input_size).to(data.device)
        c0 = torch.zeros(self.num_layers, data.size(0), self.input_size).to(data.device)
        out, _ = self.lstm(data, (h0, c0))
        out = out.transpose(1,2)
        out = self.out(out).transpose(1,2)
        return out