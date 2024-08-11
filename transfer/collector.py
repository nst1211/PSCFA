from tqdm import tqdm  # 用于显示进度条
import numpy as np
import torch

# 设置随机种子，保证结果的可重复性
torch.manual_seed(20230226)  # 设置CPU的随机种子
torch.cuda.manual_seed(20230226)  # 设置GPU的随机种子
np.random.seed(20231219)  # 设置NumPy的随机种子

def get_prototype(feature_dict):
    """
    根据特征字典生成每个标签的原型向量（即特征的平均值）。
    """
    prototype_dict = {}
    for label_idx in feature_dict.keys():
        feats = feature_dict[label_idx]
        if type(feats) == np.ndarray:  # 检查特征是否为NumPy数组
            prototype = np.mean(feats, 0)  # 计算每个标签的特征均值
            prototype_dict[label_idx] = prototype  # 将均值存入原型字典
        else:
            prototype_dict[label_idx] = None  # 如果特征为空，则对应的原型为None
    return prototype_dict


def get_head(feature_dict, args):
    """
    根据给定的阈值，将特征字典中的类别分为头部类别和尾部类别。
    """
    head_list = []
    tail_list = []
    for label_idx in feature_dict.keys():
        feats = feature_dict[label_idx]
        if type(feats) == np.ndarray:  # 检查特征是否为NumPy数组
            instance_num = len(feats)  # 获取当前标签的样本数量
            if instance_num > args.threshold2:  # 根据阈值判断是头部还是尾部
                head_list.append(label_idx)  # 加入头部类别列表
            else:
                tail_list.append(label_idx)  # 加入尾部类别列表
    return head_list, tail_list


def collect(model, train_iter, args):
    """
    收集模型在数据集上的特征表示，生成特征字典。
    """
    device = 'cuda'
    model.to(device)
    fix_model(model)  # 冻结模型参数
    model.eval()  # 设置模型为评估模式
    feature_dict = {}  # 初始化空的特征字典
    
    # 为每个标签初始化None
    for idx in range(21):
        feature_dict[idx] = None
    
    # 在不计算梯度的上下文中执行模型前向传播
    with torch.no_grad():
        # 使用tqdm显示进度条，加载每个批次的数据
        for train_data, train_length, train_label in tqdm(train_iter):
            # 将数据加载到GPU
            train_data, train_length, train_label = train_data.to(device), train_length.to(device), train_label.to(device)
            input_id = train_data
            trg = train_label
            
            # 通过模型的特征提取器获取表示向量
            representation = model.extractor(input_id).to(device)
            
            # 将当前批次的表示向量和标签追加到特征字典中
            feature_dict = dict_append_batch(representation, trg, feature_dict)
    
    # 将生成的特征字典保存到指定路径
    np.save(args.feature_dict_path, feature_dict)
    
    # 解冻模型参数
    unfix_model(model)
    model.train()  # 恢复模型到训练模式
    return feature_dict


def dict_append_batch(representation, trg, feature_dict):
    """
    将一个批次的表示向量和标签追加到特征字典中。
    """
    batch_size, label_size, hidden_size = representation.shape
    for batch_idx in range(batch_size):
        for label_idx in range(label_size):
            if trg[batch_idx, label_idx] == 1:  # 只处理正标签
                vector = torch.unsqueeze(representation[batch_idx, label_idx, :], 0)  # 添加维度使其变为(1, hidden_size)
                vector = vector.detach().cpu().numpy()  # 将张量转换为NumPy数组
                feature_dict = dict_append_each(vector, label_idx, feature_dict)  # 追加到特征字典中
    return feature_dict


def dict_append_each(vector, label_idx, feature_dict):
    """
    将单个向量追加到特征字典的对应标签下。
    """
    if feature_dict[label_idx] is None:
        feature_dict[label_idx] = vector  # 如果该标签下无特征，直接赋值
    else:
        feature_dict[label_idx] = np.vstack((feature_dict[label_idx], vector))  # 如果有特征，追加新的向量
    return feature_dict


def fix_model(model):
    """
    冻结模型的所有参数，使其在训练过程中不更新。
    """
    for param in model.parameters():
        param.requires_grad = False


def unfix_model(model):
    """
    解冻模型的所有参数，使其可以更新。
    """
    for param in model.parameters():
        param.requires_grad = True


def get_queue(feature_dict):
    """
    根据特征字典生成一个包含所有标签的原型队列。
    """
    queue = []
    for label_idx in feature_dict.keys():
        feats = feature_dict[label_idx]
        if type(feats) == np.ndarray:  # 检查特征是否为NumPy数组
            prototype = np.mean(feats, 0)  # 计算每个标签的特征均值
        else:
            prototype = np.zeros(240)  # 如果特征为空，使用全零向量代替
        queue.append(prototype)  # 将原型添加到队列中
    queue = np.array(queue)  # 转换为NumPy数组
    return queue
