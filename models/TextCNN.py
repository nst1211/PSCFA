import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 设置CPU和GPU的随机种子，保证结果的可重复性
torch.manual_seed(20230226)
torch.cuda.manual_seed(20230226)

class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, n_filters: int, filter_sizes: list, output_dim: int,
                 dropout: float):
        super(TextCNN, self).__init__()
        
        # 初始化textCNN特征提取器和分类器
        self.extractor = textCNN(vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout)
        self.clf = Classifier(filter_sizes, n_filters, output_dim)

    def forward(self, data):
        # 通过特征提取器获取表示
        representation = self.extractor(data)
        # 使用分类器对表示进行分类
        logit = self.clf(representation)
        return logit


class textCNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, n_filters: int, filter_sizes: list, output_dim: int,
                 dropout: float):
        super(textCNN, self).__init__()
        
        # 定义嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 定义多个一维卷积层，使用不同的卷积核大小（filter_sizes）
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim,
                                              out_channels=n_filters,
                                              kernel_size=fs,
                                              padding='same')  # 使用'same'填充保证输出与输入尺寸相同
                                    for fs in filter_sizes])
        
        # 定义dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 使用Mish作为激活函数
        self.Mish = nn.Mish()

    def forward(self, data, length=None):
        # 将输入数据转换为嵌入向量
        embedded = self.embedding(data)
        
        # 调整嵌入向量的维度以匹配Conv1d的输入格式 [batch_size, embedding_dim, seq_len]
        embedded = embedded.permute(0, 2, 1)
        
        # 对嵌入向量应用多个卷积层和Mish激活函数
        conved = [self.Mish(conv(embedded)) for conv in self.convs]
        
        # 对每个卷积后的结果进行最大池化，缩减每个特征图的长度
        pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 10)) for conv in conved]
        
        # 将池化后的特征图拼接在一起，并应用dropout
        cat = self.dropout(torch.cat(pooled, dim=1))
        
        # 重塑张量为 [batch_size, 21, 240] 以匹配分类器的输入
        cat = cat.reshape(-1, 21, 240)
        return cat


class Classifier(nn.Module):
    def __init__(self, filter_sizes, n_filters, output_dim):
        super(Classifier, self).__init__()
        
        # 定义全连接层的序列，逐步降低维度，最终输出类别数的logits
        self.fc = nn.Sequential(
            nn.Flatten(),  # 展平输入张量
            nn.Dropout(0.6),  # 应用dropout
            nn.Linear(len(filter_sizes) * n_filters * 10, 2560),  # 全连接层，输入尺寸与卷积层输出相关
            nn.Mish(),  # Mish激活函数
            nn.Dropout(),  # 应用dropout
            nn.Linear(2560, 512),  # 第二个全连接层
            nn.Mish(),  # Mish激活函数
            nn.Dropout(),  # 应用dropout
            nn.Linear(512, 256),  # 第三个全连接层
            nn.Mish(),  # Mish激活函数
            nn.Linear(256, output_dim)  # 最后一个全连接层，输出维度为类别数
        )

    def forward(self, representation):
        # 将表示输入到全连接层序列中，得到分类结果logits
        return self.fc(representation)
