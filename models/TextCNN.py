
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


torch.manual_seed(20230226)  # 设置CPU的随机种子
torch.cuda.manual_seed(20230226)  # 设置GPU的随机种子


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, n_filters: int, filter_sizes: list, output_dim: int,
                 dropout: float):
        super(TextCNN, self).__init__()
        self.extractor = textCNN(vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout)# 定义 extractor ，提取文本表示
        self.clf = Classifier(filter_sizes, n_filters, output_dim) # 定义clf ，用于分类，由 Classifier 实现

    def forward(self, data):# 定义前向传播函数，输入为 data
        representation = self.extractor(data)  # 使用 extractor 提取数据表示
        logit = self.clf(representation) # 将提取的表示传递给分类器，得到输出 logits
        return logit 


class textCNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, n_filters: int, filter_sizes: list, output_dim: int,
                 dropout: float):
        super(textCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 定义嵌入层
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim,
                                              out_channels=n_filters,
                                              kernel_size=fs,
                                              padding='same')
                                    for fs in filter_sizes])
        self.dropout = nn.Dropout(dropout) # 定义 Dropout 层，防止过拟合
        self.Mish = nn.Mish() # 使用 Mish 激活函数

    def forward(self, data, length=None):
        embedded = self.embedding(data) #嵌入
        embedded = embedded.permute(0, 2, 1)
        conved = [self.Mish(conv(embedded)) for conv in self.convs] #卷积
        pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 10)) for conv in conved]  # Use conved_cbam instead of conved #池化
        cat = self.dropout(torch.cat(pooled, dim=1)) #拼接
        cat = cat.reshape(-1, 21, 240)  # [batch_size, 21, 240]
        return cat

#分类器
class Classifier(nn.Module):
    def __init__(self, filter_sizes, n_filters, output_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.6),
            nn.Linear(len(filter_sizes) * n_filters * 10, 2560),
            nn.Mish(),
            nn.Dropout(),
            nn.Linear(2560, 512),
            nn.Mish(),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.Mish(),
            nn.Linear(256, output_dim) )

    def forward(self, representation):
        return self.fc(representation)

