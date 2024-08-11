import torch
import torch.nn as nn
from losses import BinaryDiceLoss  

# 定义对比学习训练函数
def train_one_batch_contrastive(model, train_data, train_label, optimizer, prototype_queue, args):
    # 使用BinaryDiceLoss作为损失函数之一
    criterion = BinaryDiceLoss()
    device = 'cuda'
    
    # 将模型移动到GPU
    model.to(device)
    
    # 将输入数据和标签移动到GPU
    input_id = train_data.to(device)
    train_label = train_label.to(device)
    
    # 优化器梯度清零
    optimizer.zero_grad()
    
    # 设置模型为训练模式
    model.train()
    
    # 通过模型提取表示向量
    representation = model.extractor(input_id).to(device)
    
    # 使用分类器对表示向量进行预测
    y_hat = model.clf(representation).to(device)
    
    # 计算损失
    loss1 = criterion(y_hat, train_label.float())  # BinaryDiceLoss
    loss2 = contrastive_loss(representation, train_label.cpu(), prototype_queue, args)  # 对比损失
    loss = (1 - args.contrastive_weight) * loss1 + args.contrastive_weight * loss2  # 综合损失
    
    # 确保loss可以进行反向传播
    loss = loss.requires_grad_(True)
    
    # 反向传播计算梯度
    loss.backward()
    
    # 更新模型参数
    optimizer.step(closure=None)
    
    # 返回当前批次的损失值
    return loss.item()


# 定义对比损失函数
def contrastive_loss(representation, traget, prototype_queue, args):
    device = 'cuda'
    ce_loss = nn.CrossEntropyLoss()  # 使用交叉熵损失
    representation = nn.functional.normalize(representation, dim=-1).to(device)  # 对表示向量进行归一化
    queue = nn.functional.normalize(torch.from_numpy(prototype_queue), dim=-1).to(device).float().detach()  # 归一化原型队列并转为tensor
    batch_queue = queue
    loss = 0
    count = 0
    
    # 获取表示向量的维度信息
    batch_size, label_size, hidden_size = representation.shape
    
    # 遍历每个批次和标签
    for batch_idx in range(batch_size):
        for label_idx in range(label_size):
            if traget[batch_idx, label_idx] == 1:  # 只对正样本计算对比损失
                q = representation[batch_idx, label_idx, :]  # 获取当前标签的表示向量
                k = get_positive_proto(queue, label_idx)  # 获取对应的正样本原型向量
                
                # 计算正样本的logit值
                l_pos = torch.einsum('c,c->', [q, k])
                l_pos = l_pos.view(1, 1)
                
                # 计算负样本的logit值
                l_neg = torch.einsum('kc,c->k', [batch_queue.clone().detach(), q])
                l_neg = l_neg.view(1, l_neg.size(0))
                
                # 将正负样本的logits拼接在一起
                logits = torch.cat([l_pos, l_neg], dim=1)
                logits /= args.T  # 应用温度缩放
                
                # 创建标签，用于交叉熵损失计算，正样本标签为0
                labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
                count += 1
                
                # 计算当前样本的交叉熵损失并累加
                temp_loss = ce_loss(logits, labels)
                loss += temp_loss
    
    # 返回平均损失，如果count为0可能会引发除零错误
    return loss / count / batch_size


# 获取对应标签的正样本原型向量
def get_positive_proto(prototype_queue, label_idx):
    return prototype_queue[label_idx]


# 从原型队列中选取与目标标签匹配的样本
def get_selected_proto(queue, target, representation):
    batch_size, label_size, hidden_size = representation.shape
    label_set = []
    
    # 遍历每个批次和标签
    for batch_idx in range(batch_size):
        for label_idx in range(label_size):
            if target[batch_idx, label_idx] == 1:  # 只选择目标标签为正的样本
                if label_idx not in label_set:  # 去重
                    label_set.append(label_idx)
    
    # 将标签集合转换为tensor并移动到GPU
    batch_index = torch.Tensor(label_set).cuda().long()
    
    # 从队列中选择对应的原型向量
    batch_queue = torch.index_select(queue, 0, batch_index)
    return batch_queue
