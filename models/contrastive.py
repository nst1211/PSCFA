import torch
import torch.nn as nn
from losses import BinaryDiceLoss




def train_one_batch_contrastive(model,train_data,train_label,optimizer,prototype_queue,args):
    criterion = BinaryDiceLoss()
    device='cuda'
    model.to(device)
    input_id = train_data.to(device)
    train_label = train_label.to(device)
    optimizer.zero_grad() #优化器的梯度值清零
    model.train() #将模型设置为训练模式
    representation = model.extractor(input_id).to(device)  # 将嵌入向量转换为表示向量
    y_hat = model.clf(representation).to(device)  # 表示向量预测为输出向量
    # 计算损失
    loss1= criterion(y_hat, train_label.float())  # 交叉熵损失
    loss2 = contrastive_loss(representation, train_label.cpu(), prototype_queue, args)
    loss =(1-args.contrastive_weight)*loss1+ args.contrastive_weight * loss2
    loss = loss.requires_grad_(True)
    loss.backward()
    optimizer.step(closure=None)
    return loss.item()


def contrastive_loss(representation, traget, prototype_queue, args):
    # mse_loss = nn.MSELoss()
    device = 'cuda'
    ce_loss = nn.CrossEntropyLoss()
    representation = nn.functional.normalize(representation, dim=-1).to(device)
    queue = nn.functional.normalize(torch.from_numpy(prototype_queue), dim=-1).to(device).float().detach()
    batch_queue = queue
    loss = 0
    count = 0
    batch_size, label_size, hidden_size = representation.shape
    for batch_idx in range(batch_size):
        for label_idx in range(label_size):
            if traget[batch_idx, label_idx] ==1:
                q = representation[batch_idx,label_idx,:] #feat_size
                k = get_positive_proto(queue, label_idx) #feat_size, label_size*feat_size

                l_pos = torch.einsum('c,c->', [q, k])
                l_pos = l_pos.view(1,1)
                # negative logits: K
                l_neg = torch.einsum('kc,c->k', [batch_queue.clone().detach(), q])
                l_neg = l_neg.view(1, l_neg.size(0))
                # logits: (1+K)
                logits = torch.cat([l_pos, l_neg], dim=1)
                logits /= args.T
                # labels: positive key indicators
                labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
                count+=1
                temp_loss = ce_loss(logits, labels)
                loss+=temp_loss
    return loss/count/batch_size

def get_positive_proto(prototype_queue, label_idx):
    # return prototype_queue[label_idx], None
    return prototype_queue[label_idx]

def get_selected_proto(queue, target, representation):
    batch_size, label_size, hidden_size = representation.shape
    label_set = []
    for batch_idx in range(batch_size):
        for label_idx in range(label_size):
            if target[batch_idx, label_idx] ==1:
                if label_idx not in label_set:
                    label_set.append(label_idx)
    batch_index = torch.Tensor(label_set).cuda().long()
    batch_queue = torch.index_select(queue, 0, batch_index)
    return batch_queue

