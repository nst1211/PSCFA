
import os

from models.TextCNN import textCNN,Classifier
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from pathlib import Path
import argparse
import torch

#命令行参数获取
def ArgsGet():
    parse = argparse.ArgumentParser(description='PSCFA')
    parse.add_argument('-file', type=str, default='./test.fasta', help='fasta file')
    parse.add_argument('-out_path', type=str, default='./result', help='output path')
    args = parse.parse_args()
    return args


def get_data(file):
    # getting file and encoding
    seqs = []
    names = []
    seq_length = []
    with open(file) as f:
        for each in f:
            if each == '\n':
                continue
            elif each[0] == '>':
                names.append(each)
            else:
                seqs.append(each.rstrip())

    # encoding
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    max_len = 50
    data_e = []
    delSeq = 0
    for i in range(len(seqs)):
        sign = True
        if len(seqs[i]) > max_len or len(seqs[i]) < 5:
            print(f'本方法只能识别序列长度在5-50AA的多肽，该序列将不能识别：{seqs[i]}')
            del names[i-delSeq]
            delSeq += 1
            continue
        length = len(seqs[i])
        seq_length.append(length)
        elemt, st = [], seqs[i]
        for j in st:
            if j == ',' or j == '1' or j == '0':
                continue
            elif j not in amino_acids:
                sign = False
                print(f'本方法只能识别包含天然氨基酸的多肽，该序列不能识别{seqs[i]}')
                del names[i-delSeq]
                delSeq += 1
                break

            index = amino_acids.index(j)
            elemt.append(index)
        if length <= max_len and sign:
            elemt += [0] * (max_len - length)
            data_e.append(elemt)

    return np.array(data_e), names, np.array(seq_length)


def load_models():
    feature_extractor = textCNN(
        vocab_size=21,  # 假设词汇表大小为10000
        embedding_dim=256,  # 嵌入维度为300
        n_filters=126,  # 卷积核数量为100
        filter_sizes=[3, 4, 5, 6],  # 卷积核大小列表
        output_dim=21,  # 假设输出类别为21
        dropout=0.6  # Dropout 概率为0.5
    )

    # 定义分类器 (Classifier)，根据实际参数进行实例化
    classifier =Classifier(
        filter_sizes=[3, 4, 5 ,6],  # 卷积核大小列表
        n_filters=126,  # 卷积核数量为100
        output_dim=21  # 假设输出类别为21
    )

    # 加载 model.pth 中的 state_dict 并提取特征提取器的参数
    state_dict = torch.load('./saved_models/model.pth')
    feature_extractor_state_dict = {k.replace('extractor.', ''): v for k, v in state_dict.items() if k.startswith('extractor')}
    feature_extractor.load_state_dict(feature_extractor_state_dict)

    # 加载 new_model.pth 中的分类器参数
    classifier.load_state_dict(torch.load('./saved_models/new_model.pth'))

    # 设置模型为评估模式
    feature_extractor.eval()
    classifier.eval()

    # 设置模型为评估模式
    feature_extractor.eval()
    classifier.eval()

    return feature_extractor, classifier


def pre_my(test_data, seq_length, output_path, names):
    # models
    # 1. 加载模型
    feature_extractor, classifier = load_models()
    # 2. 提取特征
    with torch.no_grad():
        features = feature_extractor(test_data)

    # 3. 使用分类器进行预测
    with torch.no_grad():
        score_label = classifier(features)

    for i in range(len(score_label)):
        for j in range(len(score_label[i])):
            if score_label[i][j] < 0.5:
                score_label[i][j] = 0
            else:
                score_label[i][j] = 1

    # label
    peptides = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP',
                'AVP',
                'BBP', 'BIP',
                'CPP', 'DPPIP',
                'QSP', 'SBP', 'THP']
    functions = []
    for e in score_label:
        temp = ''
        for i in range(len(e)):
            if e[i] == 1:
                temp = temp + peptides[i] + ','
            else:
                continue
        if temp == '':
            temp = 'none'
        if temp[-1] == ',':
            temp = temp.rstrip(',')
        functions.append(temp)

    output_file = os.path.join(output_path, 'result.txt')
    with open(output_file, 'w') as f:
        for i in range(len(names)):
            f.write(names[i])
            f.write('functions:' + functions[i] + '\n')


if __name__ == '__main__':
    args = ArgsGet()
    file = args.file  # fasta file
    output_path = args.out_path  # output path

    # building output path directory
    Path(output_path).mkdir(exist_ok=True)

    # reading file and encoding
    data, names, seq_length = get_data(file)
    data = torch.LongTensor(data)
    seq_length = torch.LongTensor(seq_length)

    # prediction
    pre_my(data, seq_length, output_path, names)