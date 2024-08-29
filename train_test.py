import csv
import os
import time
import torch
import random
import pandas as pd
import logging
import numpy as np
import estimate  # 导入模型评估相关的模块
import calibration  # 导入模型校准相关的模块
from losses import BinaryDiceLoss  # 导入自定义的损失函数
from models.TextCNN import TextCNN  # 导入TextCNN模型
from DataLoad import data_load  # 导入数据加载模块
from train import DataTrain, predict, CosineScheduler  # 导入训练和预测相关的模块
from transfer import collector, generator, transfer_model, transfer_data, transfer_train  # 导入特征提取和迁移学习相关的模块
from models.TextCNN import Classifier  # 导入分类器模型
from torch.optim import Adam  # 导入Adam优化器
from utils import time_since  # 导入计算运行时间的工具函数

# 配置日志记录格式和级别
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置随机种子以确保结果的可重复性
torch.manual_seed(20230226)  # 设置CPU的随机种子
torch.cuda.manual_seed(20230226)  # 设置GPU的随机种子
random.seed(20231219)
np.random.seed(20231219)

torch.backends.cudnn.deterministic = True  # 固定GPU运算方式，确保结果一致性
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择运行设备

# 定义用于保存结果的列标题
RMs = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP',
       'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP', 'AVP', 'BBP',
       'BIP', 'CPP', 'DPPIP', 'QSP', 'SBP', 'THP']

# 定义CSV文件的列标题
title1 = ['Model', "Loss", 'Aiming', 'Coverage', 'Accuracy',
          'Absolute_True', 'Absolute_False', 'RunTime',
          'Test_Time']

def spent_time(start, end):
    """
    计算并返回运行时间的分钟和秒数。
    参数:
    - start: 起始时间戳
    - end: 结束时间戳
    返回:
    - minute: 运行时间的分钟数
    - secs: 运行时间的秒数
    """
    epoch_time = end - start
    minute = int(epoch_time / 60)  # 转换为分钟
    secs = int(epoch_time - minute * 60)  # 剩余的秒数
    return minute, secs

def save_results(model_name, loss_name, start, end, test_score, title, file_path):
    """
    保存模型结果到 .csv 文件。
    参数:
    - model_name: 模型名称
    - loss_name: 损失函数名称
    - start: 开始时间戳
    - end: 结束时间戳
    - test_score: 测试集得分
    - title: CSV文件的标题
    - file_path: 保存路径
    """
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # 获取当前时间
    if len(test_score) != 21:  # 如果测试得分长度不为21，创建标准格式的内容
        content = [[model_name, loss_name,
                    '%.3f' % test_score[0],
                    '%.3f' % test_score[1],
                    '%.3f' % test_score[2],
                    '%.3f' % test_score[3],
                    '%.3f' % test_score[4],
                    '%.3f' % (end - start),
                    now]]
    else:  # 如果得分为21项，添加模型名称并写入文件
        title.append('Model')
        content1 = [f'{i:.3f}' for i in test_score]
        content1.append(model_name)
        content = [content1]

    # 检查文件是否已经存在
    if os.path.exists(file_path):
        data = pd.read_csv(file_path, header=None, encoding='gbk')
        one_line = list(data.iloc[0])
        if one_line == title:
            # 如果文件存在并且标题一致，追加内容
            with open(file_path, 'a+', newline='') as t:  # 以追加方式打开文件并写入数据
                writer = csv.writer(t)
                writer.writerows(content)
        else:
            # 如果标题不一致，则重写标题和内容
            with open(file_path, 'a+', newline='') as t:
                writer = csv.writer(t)
                writer.writerow(title)
                writer.writerows(content)
    else:  # 如果文件不存在，创建新文件并写入标题和内容
        with open(file_path, 'a+', newline='') as t:
            writer = csv.writer(t)
            writer.writerow(title)
            writer.writerows(content)

def TrainAndTest(args):
    """
    训练和测试模型的主要函数。
    参数:
    - args: 命令行参数
    """
    logger.info(f'This task is {args.task}')  # 记录任务信息
    models_file = f'result/{args.task}_models.txt'  # 模型文件路径
    Time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # 获取当前时间

    # 保存参数设置
    parse_file = f"result/{args.task}_pares.txt"
    file1 = open(parse_file, 'a')
    file1.write(Time)
    file1.write('\n')
    print(args, file=file1)  # 将参数写入文件
    file1.write('\n')
    file1.close()
    file_path = "{}/{}.csv".format('result', 'model_select')  # 结果保存路径

    # 加载数据
    logger.info('Data is loading ......（￣︶￣）↗')
    train_datasets, test_datasets, subtests, weight = data_load(batch=args.batch_size,
                                                                train_direction=args.train_direction,
                                                                test_direction=args.test_direction,
                                                                subtest=args.subtest,
                                                                CV=False)  # 加载训练数据和测试数据，并进行编码
    logger.info('Data is loaded!ヾ(≧▽≦*)o')
    test_score, aim, cov, acc, ab_true, ab_false = [], 0, 0, 0, 0, 0  # 初始化评估指标
    start_time = time.time()

    for i in range(len(train_datasets)):  # 根据训练数据的个数训练若干模型
        train_dataset = train_datasets[i]
        test_dataset = test_datasets[i]
        train_start = time.time()

        # 初始化模型
        model = TextCNN(args.vocab_size, args.embedding_size, args.filter_num, args.filter_size,
                            args.output_size, args.dropout)

        model_name = model.__class__.__name__
        title_task = f"{args.task}+{model_name}"

        # 保存模型参数设置
        model_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        file2 = open(models_file, 'a')
        file2.write(model_time)
        file2.write('\n')
        print(model, file=file2)  # 打印并保存模型类细节
        file2.write('\n')
        file2.close()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  # 初始化Adam优化器
        lr_scheduler = CosineScheduler(250, base_lr=args.learning_rate, warmup_steps=20)  # 初始化余弦退火学习率调度器

        criterion = BinaryDiceLoss()  # 使用BinaryDiceLoss作为损失函数

        loss_name = criterion.__class__.__name__
        if args.pretrained == False:  # 如果没有预训练的模型，则训练并保存模型
            logger.info(f"{model_name} is training......")
            # 初始化训练类
            Train = DataTrain(model, optimizer, criterion, lr_scheduler, device=DEVICE, args=args)
            # 训练模型
            Train.train_step(train_dataset, test_dataset, args.epochs, model_name, va=False)
            logger.info(f"{model_name} training has been completed.")
            torch.save(model.state_dict(), args.check_pt_model_path)  # 保存模型状态字典
            logger.info(f"{model_name} has been saved in {args.check_pt_model_path}.")
        else:  # 如果存在预训练模型，直接加载参数
            logger.info(f"{model_name} is Loading......")
            args.check_pt_model_path = args.pretrained_path  # 设置模型路径

        if args.FA == False:  # 如果不进行数据增强和微调，直接评估模型
            device = 'cuda'
            model.load_state_dict(torch.load(args.check_pt_model_path, map_location=device))  # 加载模型状态字典
            # 在整个测试集上评估模型
            model_predictions, true_labels = predict(model,, test_dataset, device='cuda')  # 模型预测
        else:
            '--------------------------------------------Collecting特征提取-------------------------------------------------'
            device = 'cuda'
            model.load_state_dict(torch.load(args.check_pt_model_path, map_location=device))  # 模型加载
            logger.info('The augmentation of feaature is starting.... ')
            logger.info('step1:Collecting')
            start_time = time.time()
            feature_dict = collector.collect(model, train_dataset, args)
            logger.info(f'Collected Feature Dictionary Path: {args.feature_dict_path}')
            logger.info('Time for Collecting: %.1f s' % time_since(start_time))
            prototype_dict = collector.get_prototype(feature_dict)  # 获取原型
            head_list, tail_list = collector.get_head(feature_dict, args)  # 获取头部样本
            '--------------------------------------VAE构建模型，训练以及加载--------------------------------------------------'
            vae_model = transfer_model.FeatsVAE(args)
            vae_model = vae_model.to(device='cuda')
            vae_optimizer = Adam(params=filter(lambda p: p.requires_grad, vae_model.parameters()),
                                 lr=args.vae_learning_rate)
            logger.info('Get VAE Datset')
            start_time = time.time()
            train_vae_loader, valid_vae_loader = transfer_data.get_dataset(feature_dict, head_list,
                                                                           prototype_dict, args)
            logger.info('VAE training')
            transfer_train.train(vae_model, vae_optimizer, train_vae_loader, valid_vae_loader,
                                 prototype_dict, args)
            logger.info(f'Best VAE Model Path: {args.check_pt_vae_model_path}' + '.pth')
            ' ------------------------------------Augmentation数据增强------------------------------------------------------'
            logger.info('step2:Augmentation')
            start_time = time.time()
            device = 'cuda'
            logger.info('获取微调数据集')
            vae_model.load_state_dict(torch.load(args.check_pt_vae_model_path, map_location=device))
            calibration_loader = generator.generate(vae_model, tail_list, prototype_dict, feature_dict,
                                                    args)  # 训练数据加载器
            logger.info('Time for Augmentation: %.1f s' % time_since(start_time))

            '--------------------------------------Calibration模型微调-------------------------------------------------------'
            new_model = Classifier(args.filter_size, args.filter_num, args.output_size)
            logger.info('step3:Calibration')
            start_time = time.time()
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=args.calibration_learning_rate)  # 优化器
            calibration.calibrate(model, new_model, new_optimizer, train_dataset, calibration_loader, args)
            logger.info('Time for Calibration: %.1f s' % time_since(start_time))
            logger.info(f'Best Model Path: {args.check_pt_new_model_path}')
            logger.info('Predicting with augmentation')
            device = 'cuda'
            model.load_state_dict(torch.load(args.check_pt_model_path, map_location=device))
            new_model.load_state_dict(torch.load(args.check_pt_new_model_path, map_location=device))
            model_predictions, true_labels, representation = calibration.test(model, new_model, test_dataset, args)



        test_score = estimate.evaluate(model_predictions, true_labels, threshold=args.threshold)  # 模型评估
        # 保存模型泛化性能
        test_end = time.time()
        save_results(title_task, loss_name, train_start, test_end, test_score, title1, file_path)
        # 打印评估结果
        run_time = time.time()
        m, s = spent_time(start_time, run_time)  # 运行时间
        logger.info(f"{args.task}, {model_name}'s runtime:{m}m{s}s")
        logger.info("测试集：")
        logger.info(f'aiming: {test_score[0]:.3f}')
        logger.info(f'coverage: {test_score[1]:.3f}')
        logger.info(f'accuracy: {test_score[2]:.3f}')
        logger.info(f'absolute_true: {test_score[3]:.3f}')
        logger.info(f'absolute_false: {test_score[4]:.3f}\n')



def predict(model, data, device="cuda"):
    # 模型预测
    model.to(device)
    model.eval()  # 进入评估模式
    predictions = []
    labels = []
    features=[]
    with torch.no_grad():  # 取消梯度反向传播
        for x, l, y in data:
            x = x.to(device)
            l = l.to(device)
            y = y.to(device)
            representation=model.extractor(x)
            score=model.clf(representation)
            label = torch.sigmoid(score)  # 将模型预测值映射至0-1之间
            predictions.extend(label.tolist())
            labels.extend(y.tolist())
            features.extend(representation.cpu().numpy())  # 保存特征表示，需要将其转移到 CPU 上并转换为 NumPy 数组
    return np.array(predictions), np.array(labels)



