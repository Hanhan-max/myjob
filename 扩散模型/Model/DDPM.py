import logging

import torch
import torch.nn as nn
from torch import optim
from torch.optim import optimizer
from tqdm import tqdm

from tools.getData import getOneData
from 扩散模型.Model.U_net import diffusion_TCN
from 扩散模型.Model.timeVae import Time_AE


class Diffusion:
    def __init__(self, noise_steps=500, beta_start=1e-4, beta_end=0.02, len=480, device="cuda:0"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.len_audio = len
        self.device = device

        self.beta = self.prepare_noise_schedule()
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_audio(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None,None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None,None]
        er = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * er, er

    def sample_timesteps(self, n):
        return torch.randint(low=0, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.len_audio), device=self.device)
            for i in reversed(range(1, self.noise_steps)):
                t = torch.full((x.shape[0],), i, dtype=torch.int64, device=self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t]
                alpha_hat = self.alpha_hat[t]
                beta = self.beta[t]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - (1 - alpha) / torch.sqrt(1 - alpha_hat) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = torch.clamp(x, -1, 1)
        return x

def train(vaemodel,epochs):
    path ='../../data/processe_data/yeWei.csv'
    input_len = 480
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, eval_dataloader, test_dataloader, mystand = getOneData(path,input_len,1,1,24)

    model = diffusion_TCN(in_channels=32,out_channels=32)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    diffusion = Diffusion(len=480,device=device)

    for epoch in range(epochs):
        losses_train = []
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_dataloader)
        for i, item in enumerate(pbar):
            input = item[0][:,:-1,:]
            x = vaemodel.embedding(input)
            x = vaemodel.encoder(x).transpose(1,2)
            t = diffusion.sample_timesteps(x.shape[0])
            x_t, noise = diffusion.noise_audio(x, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)  # 损失函数
            losses_train.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
        tqdm.write(f"\t Epoch {epoch + 1} / {epochs}, Loss: {(sum(losses_train) / len(losses_train)):.6f}")
        # if epoch % 20 == 0:
        #     paddle.save(model.state_dict(), f"car_models/ddpm_uncond{epoch}.pdparams")
        #     sampled_images = diffusion.sample(model, n=8)
        #
        #     for i in range(8):
        #         img = sampled_images[i].transpose([1, 2, 0])
        #         img = np.array(img).astype("uint8")
        #         plt.subplot(2, 4, i + 1)
        #         plt.imshow(img)
        #     plt.show()


if __name__ == '__main__':
    model = Time_AE(input_dim=1, embedding_dim=32, num_heads=4, d_model=256)
    state_dict = torch.load('../save_model/aoutencoder_state_dict.pth')
    # 将state_dict加载到模型中
    model.load_state_dict(state_dict)
    diffusion = Diffusion()
    train(model,20)