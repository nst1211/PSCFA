import random
import torch.utils.data as data_utils
import torch
import numpy as np
from tqdm import tqdm
torch.manual_seed(20230226)  # 设置CPU的随机种子
torch.cuda.manual_seed(20230226)  # 设置GPU的随机种子
random.seed(20231219)
np.random.seed(20231219)
def get_dataset(feature_dict, head_list, prototype_dict, args):
    x_feats = None
    y_pro = None
    for label_idx in tqdm(feature_dict.keys()):
        if label_idx in head_list:
            for instance in feature_dict[label_idx]:
                pro = prototype_dict[label_idx]
                if x_feats is None:
                    x_feats = instance #
                    y_pro = pro
                else:
                    x_feats = np.vstack((x_feats, instance))
                    y_pro = np.vstack((y_pro, pro))
    x_train, x_valid, y_train, y_valid = get_shuffle(x_feats, y_pro, args)

    train_data = data_utils.TensorDataset(torch.from_numpy(x_train).type(torch.LongTensor), torch.from_numpy(y_train).type(torch.Tensor))
    val_data = data_utils.TensorDataset(torch.from_numpy(x_valid).type(torch.LongTensor), torch.from_numpy(y_valid).type(torch.Tensor))

    train_vae_loader = data_utils.DataLoader(train_data, args.vae_batch_size, shuffle=True, drop_last=False)
    valid_vae_loader = data_utils.DataLoader(val_data, args.vae_batch_size, shuffle=True, drop_last=False)

    return train_vae_loader, valid_vae_loader

def get_shuffle(x, y, args):
    idx = list(range(x.shape[0]))
    random.shuffle(idx)
    x = x[idx]
    y = y[idx]
    split_num = int(0.9*x.shape[0])
    x_train, x_valid = x[:split_num], x[split_num:]
    y_train, y_valid = y[:split_num], y[split_num:]
    return x_train, x_valid, y_train, y_valid