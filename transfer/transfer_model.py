import numpy as np
import torch
import torch.nn as nn
from torch.distributions import uniform, normal

# 设置随机种子以确保结果的可重复性
torch.manual_seed(20230226)  # 设置CPU的随机种子
torch.cuda.manual_seed(20230226)  # 设置GPU的随机种子
np.random.seed(20230226)

class FeatsVAE(nn.Module):
    """
    定义一个特征变分自编码器(FeatsVAE)类，继承自nn.Module。
    该类用于对输入特征进行编码，生成隐变量，再通过解码器生成重建的特征。
    """
    def __init__(self, args, hidden_dim=240*4):
        super(FeatsVAE, self).__init__()
        self.input_dim = 240  # 输入特征维度
        self.latent_dim = 240  # 隐变量的维度

        # 编码器：将输入特征和属性拼接并映射到隐藏维度
        self.linear = nn.Sequential(
            nn.Dropout(0.6),  # 添加dropout层，防止过拟合
            nn.Linear(self.input_dim + self.input_dim, hidden_dim),  # 全连接层，将输入映射到隐藏维度
            nn.BatchNorm1d(hidden_dim),  # 添加批标准化层，防止梯度消失或爆炸
            nn.LeakyReLU(),  # 激活函数，使用LeakyReLU
            nn.Dropout(0.6),
            nn.Linear(hidden_dim, hidden_dim),  # 再次映射到隐藏维度
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
        )

        # 用于生成均值的全连接层
        self.linear_mu = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(hidden_dim, 240),  # 将隐藏层映射到均值
            nn.ReLU()
        )

        # 用于生成对数方差的全连接层
        self.linear_logvar = nn.Sequential(
            nn.Linear(hidden_dim, self.latent_dim),  # 将隐藏层映射到对数方差
            nn.ReLU()
        )

        # 解码器：通过生成器将隐变量和属性拼接后重建特征
        self.generator = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(480, 240),  # 拼接后的隐变量和属性输入到解码器
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(240, 240),  # 生成重建的特征
        )

        # 规范化和激活函数层
        self.ln1 = nn.LayerNorm(self.input_dim)
        self.bn1 = nn.BatchNorm1d(self.input_dim)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU()
        self.z_dist = normal.Normal(0, 1)  # 标准正态分布，用于隐变量的生成
        self.Mish = nn.Mish()  # Mish激活函数
        self.gelu = nn.GELU()  # GELU激活函数

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        使用reparameterization trick从N(mu, var)中采样隐变量。
        """
        std = torch.exp(0.5 * logvar)  # 计算标准差
        eps = torch.randn_like(std)  # 从标准正态分布中采样
        return mu + eps * std  # 返回采样的隐变量

    def forward(self, x, p):
        """
        前向传播函数，定义VAE的前向计算。
        参数:
        - x: 输入特征
        - p: 属性特征
        返回:
        - mu: 编码器生成的均值
        - logvar: 编码器生成的对数方差
        - recon_feats: 重建的特征
        """
        # 将输入特征和属性特征拼接在一起
        x = torch.cat((x, p), dim=1)
        # 通过编码器进行特征提取
        x = self.linear(x)
        # 生成均值和对数方差
        mu = self.linear_mu(x)
        logvar = self.linear_logvar(x)
        # 使用reparameterization trick生成隐变量
        latent_feats = self.reparameterize(mu, logvar)
        # 将隐变量与属性特征拼接
        concat_feats = torch.cat((latent_feats, p), dim=1)
        # 通过解码器生成重建的特征
        recon_feats = self.generator(concat_feats)
        # 对重建的特征进行Mish激活
        recon_feats = self.Mish(recon_feats)

        return mu, logvar, recon_feats  # 返回均值，对数方差和重建的特征
