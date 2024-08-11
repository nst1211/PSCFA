import torch.nn as nn
import torch
import numpy as np
import time
from losses import BinaryDiceLoss
from train import CosineScheduler

# 设置随机种子，保证结果的可重复性
torch.manual_seed(20230226)  # 设置CPU的随机种子
torch.cuda.manual_seed(20230226)  # 设置GPU的随机种子
np.random.seed(20231219)  # 设置NumPy的随机种子

def calibrate(model, new_model, new_optimizer, train_iter, calibration_loader, args):
    """
    用于校准模型的函数。通过多个epoch对模型进行微调，并记录训练损失。
    """
    num_data = len(calibration_loader.dataset)
    print(f"Number of data in calibration_loader: {num_data}")
    
    # 使用CosineScheduler进行学习率调度
    lr_scheduler = CosineScheduler(250, base_lr=args.calibration_learning_rate, warmup_steps=20)
    device = 'cuda'
    
    # 将模型和校准模型移动到GPU
    model.to(device)
    fix_model(model)  # 冻结预训练模型的参数
    model.eval()  # 设置预训练模型为评估模式
    
    new_model.to(device)
    new_model.train()  # 设置校准模型为训练模式
    steps = 1

    # 训练多个epoch
    for epoch in range(1, args.calibration_epochs + 1):
        start_time = time.time()
        train_loss = []
        
        # 遍历训练数据
        for train_data, train_length, train_label in train_iter:
            train_data, train_length, train_label = train_data.to(device), train_length.to(device), train_label.to(device)
            
            # 对单个批次进行训练并计算损失
            batch_loss = train_one_batch(model, new_model, new_optimizer, train_data, train_label, args)
            
            if batch_loss is not None and not np.isnan(batch_loss):  # 检查损失是否有效
                train_loss.append(batch_loss)
            
            # 更新学习率
            if lr_scheduler:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
                    lr_scheduler.step()
                else:
                    for param_group in new_optimizer.param_groups:
                        param_group['lr'] = lr_scheduler(steps)

        end_time = time.time()
        epoch_time = end_time - start_time
        average_train_loss = np.mean(train_loss)
        
        # 打印每个epoch的损失和运行时间
        print(f'C | Epochs: {epoch} /{args.calibration_epochs}| Train Loss: {average_train_loss: .4f} |Run time:{epoch_time:.2f}s')
        
        calibrate_loss = []
        
        # 遍历校准数据
        for i, batch in enumerate(calibration_loader, 1):
            batch_loss = calibrate_one_batch(batch, new_model, new_optimizer, args)
            if batch_loss is not None and not np.isnan(batch_loss):  # 检查损失是否有效
                calibrate_loss.append(batch_loss)
            
            # 更新学习率
            if lr_scheduler:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
                    lr_scheduler.step()
                else:
                    for param_group in new_optimizer.param_groups:
                        param_group['lr'] = lr_scheduler(steps)

        end_time = time.time()
        steps += 1
    
    # 保存校准后的模型
    torch.save(new_model.state_dict(), args.check_pt_new_model_path)


def test(model, new_model, data, args):
    """
    用于测试模型的函数。对输入数据进行预测，并返回预测结果、标签和表示向量。
    """
    device = 'cuda'
    
    # 将模型移动到GPU并设置为评估模式
    model.to(device)
    model.eval()
    new_model.to(device)
    new_model.eval()
    
    predictions = []
    labels = []
    
    with torch.no_grad():  # 禁用梯度计算以加快计算速度
        for x, l, y in data:
            x = x.to(device)
            l = l.to(device)
            y = y.to(device)
            
            # 通过预训练模型提取表示向量
            representation = model.extractor(x).to(device)
            
            # 使用校准模型进行预测
            score = new_model(representation)
            
            # 将预测结果映射到0-1之间
            label = torch.sigmoid(score)
            
            # 收集预测结果和标签
            predictions.extend(label.tolist())
            labels.extend(y.tolist())
            
            # 将表示向量转换为NumPy数组
            representation = np.array(representation.cpu())
    
    return np.array(predictions), np.array(labels), representation


def calibrate_one_batch(batch, new_model, new_optimizer, args):
    """
    对单个批次进行校准的函数。计算损失并更新校准模型的参数。
    """
    device = 'cuda'
    new_model.to(device)
    
    # 获取批次数据
    representation, label_idx = batch
    representation = torch.unsqueeze(representation, 1).to(device)  # 扩展维度为 (batch_size, 1, hidden_size)
    representation = representation.repeat(1, 21, 1).detach()  # 重复表示向量以匹配标签大小 (batch_size, label_size, hidden_size)
    label_idx = label_idx.to(device)
    
    new_optimizer.zero_grad()
    new_model.train()
    
    # 使用校准模型进行预测
    y_pred = new_model(representation)
    
    # 计算校准损失
    loss = calibration_loss(y_pred, label_idx, args)
    loss.backward()
    new_optimizer.step(closure=None)
    
    return loss.item()


def calibration_loss(y_pred, label_idx, args):
    """
    计算校准损失的函数。使用二分类交叉熵损失函数。
    """
    trg = torch.sigmoid(y_pred).detach()  # 使用sigmoid将预测结果映射到0-1之间，并且不计算梯度
    for i, idx in enumerate(label_idx):
        trg[i, int(idx)] = 1  # 将目标标签位置的值设为1
    
    # 使用二分类交叉熵损失函数
    criteria = nn.BCEWithLogitsLoss()
    loss = criteria(y_pred, trg.float())
    return loss


def train_one_batch(model, new_model, new_optimizer, train_data, train_label, args):
    """
    对单个批次进行训练的函数。计算损失并更新校准模型的参数。
    """
    criterion = BinaryDiceLoss()
    device = 'cuda'
    model.to(device)
    
    # 将输入数据和标签移动到GPU
    input_id = train_data.to(device)
    train_label = train_label.to(device)
    
    new_optimizer.zero_grad()
    model.train()
    
    # 提取表示向量并使用校准模型进行预测
    representation = model.extractor(input_id).to(device)
    y_hat = new_model(representation)
    
    # 计算损失
    loss = criterion(y_hat, train_label.float())
    loss.backward()
    new_optimizer.step(closure=None)
    
    return loss.item()


def fix_model(model):
    """
    冻结模型的所有参数，使其在训练过程中不更新。
    """
    for param in model.parameters():
        param.requires_grad = False
