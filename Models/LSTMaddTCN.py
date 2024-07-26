import torch
import torch.nn as nn

class myLSTM(nn.Module):
    def __init__(self,input_dim,hidden_dim,input_len,output_len,num_layer=3):
        super(myLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,hidden_size=input_dim,num_layers=num_layer, batch_first=True)
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.num_layers = num_layer
        self.out = nn.Linear(input_len,output_len)
    def forward(self,data):
        h0 = torch.zeros(self.num_layers, data.size(0), self.input_size).to(data.device)
        c0 = torch.zeros(self.num_layers, data.size(0), self.input_size).to(data.device)
        out, _ = self.lstm(data, (h0, c0))
        out = out.transpose(1,2)
        out = self.out(out).transpose(1,2)
        return out

class myTCN(nn.Module):
    def __init__(self,input_dim,hidden_dim,input_len,output_len,dropout=0.1):
        super(myTCN, self).__init__()
        self.tcn_1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        self.tcn_2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        self.tcn_3 = nn.Conv1d(in_channels=hidden_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(input_len, output_len)

    def forward(self,input):
        input = input.transpose(1,2)
        x = self.dropout(self.relu(self.tcn_1(input)))
        # x = self.dropout(self.relu(self.tcn_2(x)))
        x = self.dropout(self.relu(self.tcn_3(x)))
        out = self.out(x).transpose(1,2)
        return out

class TandL(nn.Module):
    def __init__(self,input_dim,hidden_dim,input_len,output_len):
        super(TandL, self).__init__()

        self.lstm = myLSTM(input_dim,hidden_dim,input_len,output_len)
        self.tcn = myTCN(input_dim,hidden_dim,input_len,output_len,dropout=0.1)
        self.fc = nn.Linear(2,1)
    def forward(self,input):
        x_1 = self.lstm(input)
        x_2 = self.tcn(input)
        out = torch.cat([x_1,x_2],dim=2)
        out = self.fc(out)
        return out