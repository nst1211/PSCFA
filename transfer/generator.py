import numpy as np
import torch

from torch.distributions import  normal
import torch.utils.data as data_utils

torch.manual_seed(20230226)  # 设置CPU的随机种子
torch.cuda.manual_seed(20230226)  # 设置GPU的随机种子
np.random.seed(20231219)


def generate(vae_model, tail_list, prototype_dict, feature_dict, args):
    x_da = None
    y_da = None
    device='cuda'
    vae_model.eval()
    z_dist = normal.Normal(0, 1)
    for label_idx in prototype_dict.keys():
        if label_idx in tail_list:
            prototype = prototype_dict[label_idx]
            if type(prototype)==np.ndarray:
                da_number = args.da_number
                p = torch.from_numpy(prototype).float().to(device)
                p = p.repeat(da_number, 1)#
                Z = z_dist.sample((da_number, 240)).to(device) #
                concat_feats = torch.cat((Z, p), dim=1)
                feats = vae_model.generator(concat_feats)
                feats = vae_model.relu(vae_model.bn1(feats))
                label = torch.Tensor([label_idx]).repeat(da_number, 1)
                if x_da is None:
                    x_da = feats
                    y_da = label
                else:
                    x_da = torch.cat((x_da, feats), 0)
                    y_da = torch.cat((y_da, label), 0)
    train_data = data_utils.TensorDataset(x_da, y_da)
    #创建了一个训练数据加载器 train_loader
    train_loader = data_utils.DataLoader(train_data, args.calibration_batch_size, shuffle=True, drop_last=False)
    return train_loader

def save_da_data(vae_model, prototype_dict, feature_dict, tail_list, args):
    da_dict = {}
    tail_list = []
    vae_model.eval()
    z_dist = normal.Normal(0, 1)
    for idx in range(args.label_size):
        da_dict[idx] = None
    for label_idx in prototype_dict.keys():
        if label_idx in tail_list:
            prototype = prototype_dict[label_idx]
            if type(prototype)==np.ndarray: #not None
                original_num = feature_dict[label_idx].shape[0]
                da_number = args.da_times*original_num
                p = torch.from_numpy(prototype).float().to(args.device)
                p = p.repeat(da_number, 1)
                Z = z_dist.sample((da_number, args.feat_size)).to(args.device)
                concat_feats = torch.cat((Z, p), dim=1)
                feats = vae_model.generator(concat_feats)
                feats = vae_model.relu(vae_model.bn1(feats))
                da_dict[label_idx] = feats
    np.save(args.da_dict_path, da_dict)
    print(f'da_dict saved to {args.da_dict_path}')

