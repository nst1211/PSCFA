import torch
import torch.nn as nn
from losses import BinaryDiceLoss




def train_one_batch_contrastive(model,train_data,train_label,optimizer,prototype_queue,args):
    criterion = BinaryDiceLoss()  #损失
    device='cuda' #设备
    model.to(device)
    input_id = train_data.to(device) #样本
    train_label = train_label.to(device) #标签
    optimizer.zero_grad() #优化器的梯度值清零
    model.train() #将模型设置为训练模式
    representation = model.extractor(input_id).to(device)  # 将嵌入向量转换为表示向量
    y_hat = model.clf(representation).to(device)  # 表示向量预测为输出向量
    # 计算损失
    loss1= criterion(y_hat, train_label.float())  # 交叉熵损失
    loss2 = contrastive_loss(representation, train_label.cpu(), prototype_queue, args) #对比损失
    loss =(1-args.contrastive_weight)*loss1+ args.contrastive_weight * loss2 #总损失
    loss = loss.requires_grad_(True)
    loss.backward() 
    optimizer.step(closure=None)
    return loss.item()

#对比损失
def contrastive_loss(representation, traget, prototype_queue, args):
    # mse_loss = nn.MSELoss()
    device = 'cuda' # 定义要使用的设备为GPU
    ce_loss = nn.CrossEntropyLoss() # 定义交叉熵损失函数
    representation = nn.functional.normalize(representation, dim=-1).to(device) # 对输入的表示进行归一化处理，并将其移动到GPU上
    queue = nn.functional.normalize(torch.from_numpy(prototype_queue), dim=-1).to(device).float().detach() # 将queue赋值给batch_queue
    batch_queue = queue
    loss = 0 # 初始化
    count = 0
    batch_size, label_size, hidden_size = representation.shape # 获取表示张量的维度信息
    for batch_idx in range(batch_size): # 遍历每个批次中的样本
        for label_idx in range(label_size): # 遍历每个标签
            if traget[batch_idx, label_idx] ==1: # 如果当前样本的当前标签为1
                q = representation[batch_idx,label_idx,:] #feat_size # 提取当前样本和标签的特征表示向量
                k = get_positive_proto(queue, label_idx) #feat_size, label_size*feat_size # 获取该标签的原型向量
                # 计算正样本的logit，即q和k的点积
                l_pos = torch.einsum('c,c->', [q, k])
                l_pos = l_pos.view(1,1)
                # 计算负样本的logits，即q与整个队列batch_queue中每个向量的点积
                # negative logits: K
                l_neg = torch.einsum('kc,c->k', [batch_queue.clone().detach(), q])
                l_neg = l_neg.view(1, l_neg.size(0))
                # logits: (1+K)
                logits = torch.cat([l_pos, l_neg], dim=1)
                # 对logits进行温度缩放
                logits /= args.T
                # labels: positive key indicators
                labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
                count+=1
                temp_loss = ce_loss(logits, labels)
                # 将当前损失累加到总损失
                loss+=temp_loss
    return loss/count/batch_size
#提取对应标签的原型向量
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

