import numpy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from Models.LSTM import myLSTM
from Models.LSTMaddTCN import TandL
from Models.Mamba import Mamba
from Models.MyInformer import myInformer
from Models.MyMseLoss import MyMseLoss
from Models.Timeformer import Timeformer
from Models.Transformer import TimeTransform
from fit.train import trainer, test, decoder_test, decoder_trainer
from tools.getData import getOneData, getEDData
from tools.tools import getWindows

path = '../Informer2020/data/yeWei/yeWei.csv'
input_len = 480
decode_len = 120
predicte_len = 480
step_len = 1
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
print("输入长度:", input_len, "预测长度:", predicte_len, "滑动步长:", step_len)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



train_dataloader,eval_dataloader,test_dataloader,mystand = getOneData(path=path,input_len=input_len,predicte_len=predicte_len,step_len=step_len,batch_size=128,isstn =True)
# train_dataloader,eval_dataloader,test_dataloader,mystand = getEDData(path=path,input_len=input_len,predicte_len=predicte_len,step_len=step_len,batch_size=128,isstn =True)
# model = myInformer(input_dim = 1,embed_dim = 64,num_heads=1,head_dim=64,attention_dim =64,ff_dim=64,num_layers=3,de_num_layers=2)
model = Mamba(L =input_len,D=1,N=64,laryer_num=1,out_len=predicte_len)
# model = TandL(input_dim=1,hidden_dim=16,input_len=input_len,output_len=predicte_len)
# model = myLSTM(input_dim=1,hidden_dim=16,num_layer=4)
# model = TimeTransform(input_dim=1,num_heads=2,head_dim=64,attention_dim =8,ff_dim = 16,input_len=input_len,output_len=predicte_len,dropout=0.1)
# model = Timeformer(input_dim=1,num_heads=6,head_dim=128,attention_dim =16,ff_dim = 64,input_len=input_len,output_len=predicte_len,num_layers=6,dropout=0.1)
# model = Mamba(L=input_len,D=1,N=32,laryer_num=1,out_len=predicte_len)

model = model.to(device)
criterion = MyMseLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

## 非decoder训练
# num_epochs = 40
# trainer(num_epochs, train_dataloader, eval_dataloader,model,criterion,optimizer,input_len,device)
# result_np,true_np = test(model,test_dataloader,input_len,device)
## decoder训练
num_epochs = 10
trainer(num_epochs, train_dataloader, eval_dataloader,model,criterion,optimizer,input_len,device)
result_np,true_np = test(model,test_dataloader,input_len,device)

if mystand != None:
    result_np = mystand.inverse_transform(result_np)
    true_np = mystand.inverse_transform(true_np)
numpy.save('out/yeWei_result_withnoNorm_TCN&LSTM3.npy',result_np)
numpy.save('out/yeWei_true_withnoNorm.npy',true_np)
