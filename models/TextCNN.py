
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
        self.extractor = textCNN(vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout)
        self.clf = Classifier(filter_sizes, n_filters, output_dim)

    def forward(self, data):
        representation = self.extractor(data)
        logit = self.clf(representation)
        return logit


class textCNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, n_filters: int, filter_sizes: list, output_dim: int,
                 dropout: float):
        super(textCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim,
                                              out_channels=n_filters,
                                              kernel_size=fs,
                                              padding='same')
                                    for fs in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.Mish = nn.Mish()

    def forward(self, data, length=None):
        embedded = self.embedding(data)
        embedded = embedded.permute(0, 2, 1)
        conved = [self.Mish(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 10)) for conv in conved]  # Use conved_cbam instead of conved
        cat = self.dropout(torch.cat(pooled, dim=1))
        cat = cat.reshape(-1, 21, 240)  # [batch_size, 21, 240]
        return cat


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

