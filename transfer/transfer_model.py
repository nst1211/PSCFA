
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import uniform, normal

torch.manual_seed(20230226)  # 设置CPU的随机种子
torch.cuda.manual_seed(20230226)  # 设置GPU的随机种子
np.random.seed(20230226)

#
class FeatsVAE(nn.Module):
    def __init__(self, args, hidden_dim=240*4):
        super(FeatsVAE, self).__init__()
        self.input_dim = 240 #300
        self.latent_dim = 240
        # 编码器：将输入特征和属性拼接并映射到隐藏维度
        self.linear = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(self.input_dim +self.input_dim,hidden_dim),#240*2，240*4
            nn.BatchNorm1d(hidden_dim),  # 添加批标准化层
            nn.LeakyReLU(),
            nn.Dropout(0.6),
            nn.Linear(hidden_dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 添加批标准化层
            nn.LeakyReLU(),

        )
        self.linear_mu =  nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(hidden_dim, 240), #512*2,5120
            nn.ReLU()
        )

        self.linear_logvar =  nn.Sequential(
            nn.Linear(hidden_dim, self.latent_dim),#512,5120
            nn.ReLU()
        )
        self.generator = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(480, 240), #5120*2,512
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(240, 240), #512,512*10
        )
        self.ln1 = nn.LayerNorm(self.input_dim)
        self.bn1 = nn.BatchNorm1d(self.input_dim)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu =nn.LeakyReLU()
        self.z_dist = normal.Normal(0, 1)
        self.Mish = nn.Mish()
        self.gelu=nn.GELU()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, p):
        x = torch.cat((x, p), dim=1)
        x = self.linear(x)
        mu = self.linear_mu(x)
        logvar = self.linear_logvar(x)
        latent_feats = self.reparameterize(mu, logvar)
        concat_feats = torch.cat((latent_feats, p), dim=1) #5120*2
        recon_feats = self.generator(concat_feats)
        recon_feats = self.Mish(recon_feats)


        return mu, logvar, recon_feats





