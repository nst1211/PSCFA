import time
import torch,random
import math
import numpy as np
from torch.optim import lr_scheduler
from estimate import evaluate
from models.contrastive import train_one_batch_contrastive
from transfer import collector
import logging
torch.manual_seed(20230226)  # 设置CPU的随机种子
torch.cuda.manual_seed(20230226)  # 设置GPU的随机种子
random.seed(20231219)
np.random.seed(20231219)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

title1 = ['Model', "Loss", 'Aiming', 'Coverage', 'Accuracy',
          'Absolute_True', 'Absolute_False', 'RunTime',
          'Test_Time']



class DataTrain:
    # 训练模型
    def __init__(self, model, optimizer, criterion, scheduler=None, device="cuda",args=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = scheduler
        self.device = device
        self.args = args

    def train_step(self, train_iter, test_iter=None, epochs=None, model_name='', va=True):
        steps = 1
        for epoch in range(1, epochs + 1):  # 迭代训练模型
            start_time = time.time()
            feature_dict = collector.collect(self.model, train_iter, self.args)
            prototype_queue = collector.get_queue(feature_dict)
            train_loss = []
            for train_data, train_length, train_label in train_iter:  # 加载批量数据,train_length是torch.Size([256])每个样本序列处理前的的实际长度
                batch_loss = train_one_batch_contrastive(self.model,train_data,train_label, self.optimizer,
                                                             prototype_queue, self.args)
                train_loss.append(batch_loss)
                if self.lr_scheduler:
                    if self.lr_scheduler.__module__ == lr_scheduler.__name__:
                        # Using PyTorch In-Built scheduler
                         self.lr_scheduler.step()
                    else:
                         # Using custom defined scheduler
                         for param_group in self.optimizer.param_groups:
                             param_group['lr'] = self.lr_scheduler(steps)
            steps += 1
            end_time = time.time()
            epoch_time = end_time - start_time
            # 输出迭代结果
            print(f'{model_name}|Epoch:{epoch:003}/{epochs}|Run time:{epoch_time:.2f}s')
            print(f'Train loss:{np.mean(train_loss)}')

            if va==True:
                predictions, true_label = predict(self.model, train_iter)
                train_score = evaluate(predictions, true_label)
                print("训练集：")
                print(f'aiming: {train_score[0]:.3f}')
                print(f'coverage: {train_score[1]:.3f}')
                print(f'accuracy: {train_score[2]:.3f}')
                print(f'absolute_true: {train_score[3]:.3f}')
                print(f'absolute_false: {train_score[4]:.3f}\n')
                predictions, true_label = predict(self.model, test_iter)
                test_score = evaluate(predictions, true_label)

                print("测试集：")
                print(f'aiming: {test_score[0]:.3f}')
                print(f'coverage: {test_score[1]:.3f}')
                print(f'accuracy: {test_score[2]:.3f}')
                print(f'absolute_true: {test_score[3]:.3f}')
                print(f'absolute_false: {test_score[4]:.3f}\n')


def save_results(model_name, loss_name, start, end, test_score, title, file_path):
    # 保存模型结果 .csv文件
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if len(test_score) != 21:
        content = [[model_name, loss_name,
                    '%.3f' % test_score[0],
                    '%.3f' % test_score[1],
                    '%.3f' % test_score[2],
                    '%.3f' % test_score[3],
                    '%.3f' % test_score[4],
                    '%.3f' % (end - start),
                    now]]
    else:
        title.append('Model')
        content1 = [f'{i:.3f}' for i in test_score]
        content1.append(model_name)
        content = [content1]

    if os.path.exists(file_path):
        data = pd.read_csv(file_path, header=None, encoding='gbk')
        one_line = list(data.iloc[0])
        if one_line == title:
            with open(file_path, 'a+', newline='') as t:  # newline用来控制空的行数
                writer = csv.writer(t)  # 创建一个csv的写入器
                writer.writerows(content)  # 写入数据
        else:
            with open(file_path, 'a+', newline='') as t:
                writer = csv.writer(t)
                writer.writerow(title)  # 写入标题
                writer.writerows(content)
    else:
        with open(file_path, 'a+', newline='') as t:
            writer = csv.writer(t)
            writer.writerow(title)
            writer.writerows(content)

def test(model, new_model, data):
    device='cuda'
    model.to(device)
    model.eval()
    new_model.to(device)
    new_model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for x, l, y in data:
            x = x.to(device)
            l = l.to(device)
            y = y.to(device)
            representation = model.extractor(x).to(device)
            score = new_model(representation)
            label = torch.sigmoid(score)  # 将模型预测值映射至0-1之间
            predictions.extend(label.tolist())
            labels.extend(y.tolist())
        return np.array(predictions), np.array(labels)



def predict(model, data, device="cuda"):
    # 模型预测
    model.to(device)
    model.eval()  # 进入评估模式
    predictions = []
    labels = []

    with torch.no_grad():  # 取消梯度反向传播
        for x, l, y in data:
            x = x.to(device)
            l = l.to(device)
            y = y.to(device)
            score = model(x)

            label = torch.sigmoid(score)  # 将模型预测值映射至0-1之间

            predictions.extend(label.tolist())
            labels.extend(y.tolist())

    return np.array(predictions), np.array(labels)


class CosineScheduler:
    # 退化学习率
    def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch - 1) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
                           (1 + math.cos(math.pi * (epoch - 1 - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr



