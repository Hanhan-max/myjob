import numpy
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from tqdm import tqdm

from Models.MyMseLoss import MyMseLoss
from tools.getData import getOneData
from 扩散模型.Model.timeVae import timeVae, Time_AE

path ='../../data/processe_data/yeWei.csv'
input_len = 480
train_dataloader, eval_dataloader, test_dataloader, mystand = getOneData(path,input_len,1,1,58)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

model = Time_AE(input_dim=1,embedding_dim=32,num_heads=4,d_model=256)

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

criterion = MyMseLoss()

epochs = 10
for epoch in tqdm(range(epochs)):
    model.train()
    losses_train = []
    for i, item in enumerate(train_dataloader):
        input = item[0][:, :input_len].to(device)
        traget = item[0][:, :input_len].to(device)
        out= model(input)
        loss = criterion(out, traget)
        losses_train.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tqdm.write(f"\t Epoch {epoch + 1} / {epochs}, Loss: {(sum(losses_train) / len(losses_train)):.6f}")
    if (epoch + 1) % 2 == 0:
        model.eval()
        with torch.no_grad():
            losses = []
            for i, item in enumerate(eval_dataloader):
                input = item[0][:, :input_len].to(device)
                traget = item[0][:, :input_len].to(device)
                out = model(input)
                loss = criterion(out, traget)
                losses.append(loss)
            print(f'\nEpoch [{epoch + 1}/{epochs}], Eval_Loss: {sum(losses) / len(losses) :.4f}')

result = []

true = []
for i, item in enumerate(test_dataloader):
    input = item[0][:, :input_len].to(device)
    traget = item[0][:, :input_len]
    out = model(input)
    result.append(out.detach().cpu().numpy())
    true.append(traget)
result_np = np.concatenate(result, axis=0)
true_np = np.concatenate(true, axis=0)

if mystand != None:
    result_np = mystand.inverse_transform(result_np)
    true_np = mystand.inverse_transform(true_np)
numpy.save('../out/yeWei_AE.npy',result_np)
numpy.save('../out/yeWei_true_AE.npy',true_np)
torch.save(model.state_dict(), '../save_model/aoutencoder_state_dict.pth')