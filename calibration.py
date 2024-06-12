
import torch.nn as nn
import torch
import numpy as np
import time
from losses import BinaryDiceLoss
from train import CosineScheduler
torch.manual_seed(20230226)  # 设置CPU的随机种子
torch.cuda.manual_seed(20230226)  # 设置GPU的随机种子

np.random.seed(20231219)


def calibrate(model, new_model, new_optimizer, train_iter, calibration_loader, args):
    num_data = len(calibration_loader.dataset)
    print(f"Number of data in calibration_loader: {num_data}")
    lr_scheduler = CosineScheduler(250, base_lr=args.calibration_learning_rate, warmup_steps=20)
    device = 'cuda'
    model.to(device)
    fix_model(model)
    model.eval()
    new_model.to(device)
    new_model.train()
    steps = 1

    for epoch in range(1, args.calibration_epochs + 1):  # 迭代训练模型
        start_time = time.time()
        train_loss = []
        for train_data, train_length, train_label in train_iter:
            train_data, train_length, train_label = train_data.to(device), train_length.to(device), train_label.to(
                device)
            batch_loss = train_one_batch(model, new_model, new_optimizer, train_data, train_label, args)
            if batch_loss is not None and not np.isnan(batch_loss):  # 检查损失是否有效
                train_loss.append(batch_loss)
            if lr_scheduler:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
                    lr_scheduler.step()
                else:
                    for param_group in new_optimizer.param_groups:
                        param_group['lr'] = lr_scheduler(steps)

        end_time = time.time()
        epoch_time = end_time - start_time
        average_train_loss = np.mean(train_loss)
        print(
            f'C | Epochs: {epoch} /{args.calibration_epochs}| Train Loss: {average_train_loss: .4f} |Run time:{epoch_time:.2f}s')
        calibrate_loss = []
        for i, batch in enumerate(calibration_loader, 1):
            batch_loss = calibrate_one_batch(batch, new_model, new_optimizer, args)
            if batch_loss is not None and not np.isnan(batch_loss):  # 检查损失是否有效
                calibrate_loss.append(batch_loss)
            if lr_scheduler:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
                    lr_scheduler.step()
                else:
                    for param_group in new_optimizer.param_groups:
                        param_group['lr'] = lr_scheduler(steps)

        end_time = time.time()
        steps += 1
    torch.save(new_model.state_dict(),args.check_pt_new_model_path)



def test(model, new_model,data,args):
    # 模型预测
    device='cuda'
    model.to(device)
    model.eval()  # 进入评估模式
    new_model.to(device)
    new_model.eval()
    predictions = []
    labels = []
    with torch.no_grad(): # 取消梯度反向传播
        for x, l, y in data:
            x = x.to(device)
            l = l.to(device)
            y = y.to(device)
            representation = model.extractor(x).to(device)
            score = new_model(representation)
            label = torch.sigmoid(score)  # 将模型预测值映射至0-1之间
            predictions.extend(label.tolist())
            labels.extend(y.tolist())
            representation = np.array(representation.cpu())
    return np.array(predictions), np.array(labels), representation



def calibrate_one_batch(batch, new_model, new_optimizer, args):
    device='cuda'
    new_model.to(device)
    representation, label_idx = batch
    representation = torch.unsqueeze(representation, 1).to(device) # batch_size*1*hidden_size
    representation = representation.repeat(1, 21, 1).detach() # batch_size*label_size*hidden_size
    label_idx = label_idx.to(device)
    new_optimizer.zero_grad()
    new_model.train()
    y_pred = new_model(representation)
    loss = calibration_loss(y_pred, label_idx,args)
    loss.backward()
    new_optimizer.step(closure=None)
    return loss.item()

def calibration_loss(y_pred, label_idx,args):
    trg = torch.sigmoid(y_pred).detach() #without gradient
    for i, idx in enumerate(label_idx):
        trg[i,int(idx)] = 1
    criteria = nn.BCEWithLogitsLoss()
    loss = criteria(y_pred, trg.float())
    return loss

def train_one_batch(model, new_model,new_optimizer,train_data, train_label,  args):
    criterion = BinaryDiceLoss()
    # train for one batch
    device = 'cuda'
    model.to(device)
    input_id = train_data.to(device)
    train_label = train_label.to(device)
    new_optimizer.zero_grad()
    model.train()
    representation=model.extractor(input_id).to(device)
    y_hat = new_model(representation)
    # 计算损失
    loss = criterion(y_hat, train_label.float()) # 交叉熵损失
    loss.backward()
    new_optimizer.step(closure=None)
    return loss.item()


def fix_model(model):
    for param in model.parameters():
        param.requires_grad = False
