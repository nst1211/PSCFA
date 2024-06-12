from tqdm import tqdm
import numpy as np
import torch
torch.set_printoptions(threshold=np.inf)

torch.manual_seed(20230226)  # 设置CPU的随机种子
torch.cuda.manual_seed(20230226)  # 设置GPU的随机种子

np.random.seed(20231219)


def get_prototype(feature_dict):
    prototype_dict = {}
    for label_idx in feature_dict.keys():
        feats = feature_dict[label_idx]
        if type(feats) == np.ndarray:
            prototype = np.mean(feature_dict[label_idx], 0)
            prototype_dict[label_idx] = prototype
        else:
            prototype_dict[label_idx] = None
    return prototype_dict




def get_head(feature_dict,args): #根据给定的阈值，将特征字典中的类别分为头部类别和尾部类别两个列表
    head_list = []
    tail_list = []
    for label_idx in feature_dict.keys():
        feats = feature_dict[label_idx]
        if type(feats) == np.ndarray:
            instance_num = len(feats)
            if instance_num > args.threshold2:
                head_list.append(label_idx)
            else:
                tail_list.append(label_idx)
    return head_list, tail_list


def collect(model,train_iter,args):  #特征字典
    device='cuda'
    model.to(device)
    fix_model(model)
    model.eval()
    feature_dict = {}#创建空的特征字典
    for idx in range(21):
        feature_dict[idx] = None
    with torch.no_grad():
        for train_data, train_length, train_label in tqdm(train_iter):  # 加载批量数据#迭代train_loader中的每一个批次，并且通过tqdm库添加进度条
            # 使数据与模型在同一设备中
            train_data, train_length, train_label = train_data.to(device), train_length.to(device), train_label.to(
                device)
            input_id = train_data
            trg = train_label
            representation= model.extractor(input_id).to(device)
            feature_dict = dict_append_batch(representation, trg, feature_dict)
    np.save(args.feature_dict_path, feature_dict)
    unfix_model(model)
    model.train()
    return feature_dict

def dict_append_batch(representation, trg, feature_dict):
    batch_size,label_size, hidden_size = representation.shape
    for batch_idx in range(batch_size):
        for label_idx in range(label_size):
            if trg[batch_idx, label_idx] ==1:
                vector = torch.unsqueeze(representation[batch_idx,label_idx,:], 0) #1*hidden_size
                vector = vector.detach().cpu().numpy()
                feature_dict = dict_append_each(vector, label_idx, feature_dict)
    return feature_dict


def dict_append_each(vector, label_idx, feature_dict):
    if feature_dict[label_idx] is None:
        feature_dict[label_idx] = vector
    else:
        feature_dict[label_idx] = np.vstack((feature_dict[label_idx], vector)) #torch.cat((feature_dict[label_idx], vector), 0)
    return feature_dict

def fix_model(model):
    for param in model.parameters():
        param.requires_grad = False

def unfix_model(model):
    for param in model.parameters():
        param.requires_grad = True


def get_queue(feature_dict):
    queue = []
    for label_idx in feature_dict.keys():
        feats = feature_dict[label_idx]
        if type(feats) == np.ndarray:
            prototype = np.mean(feature_dict[label_idx], 0)#1*feat_size
        else:
            prototype = np.zeros(240) #zero for the lack label
        queue.append(prototype)
    queue = np.array(queue)
    return queue
