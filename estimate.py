#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 10:28
# @Author  : ywh
# @File    : estimate.py
# @Software: PyCharm

# 多标签评价指标

def Aiming(y_hat, y):
    """
    Aiming (Precision) 是反映预测标签中正确标签的比例，衡量预测标签中有多少命中了真实标签。

    参数:
    y_hat: 预测标签矩阵，shape为(n, m)，n表示样本数量，m表示标签数量
    y: 真实标签矩阵，shape与y_hat相同

    返回:
    aiming: 精确率的平均值
    """

    n, m = y_hat.shape  # n表示样本数量，m表示标签数量

    score_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:  # 计算 L ∪ L* （预测标签与真实标签的并集）
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:  # 计算 L ∩ L* （预测标签与真实标签的交集）
                intersection += 1
        if intersection == 0:  # 如果没有交集，则跳过该样本
            continue
        score_k += intersection / sum(y_hat[v])  # 计算精确率
    return score_k / n  # 返回所有样本的平均精确率


def Coverage(y_hat, y):
    """
    Coverage (Recall) 是反映真实标签中被正确预测的标签比例，衡量真实标签中有多少被预测命中。

    参数:
    y_hat: 预测标签矩阵，shape为(n, m)
    y: 真实标签矩阵，shape与y_hat相同

    返回:
    coverage: 召回率的平均值
    """

    n, m = y_hat.shape

    score_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        score_k += intersection / sum(y[v])  # 计算召回率

    return score_k / n  # 返回所有样本的平均召回率


def Accuracy(y_hat, y):
    """
    Accuracy (准确率) 是反映正确预测标签在总标签中的比例，包括正确和错误预测的标签以及漏预测的真实标签。

    参数:
    y_hat: 预测标签矩阵，shape为(n, m)
    y: 真实标签矩阵，shape与y_hat相同

    返回:
    accuracy: 准确率的平均值
    """

    n, m = y_hat.shape

    score_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        score_k += intersection / union  # 计算准确率
    return score_k / n  # 返回所有样本的平均准确率


def AbsoluteTrue(y_hat, y):
    """
    AbsoluteTrue 是计算预测标签与真实标签完全相同的样本比例。

    参数:
    y_hat: 预测标签矩阵，shape为(n, m)
    y: 真实标签矩阵，shape与y_hat相同

    返回:
    absolute_true: 完全正确预测的样本比例
    """

    n, m = y_hat.shape
    score_k = 0
    for v in range(n):
        if list(y_hat[v]) == list(y[v]):  # 如果预测标签与真实标签完全一致
            score_k += 1
    return score_k / n  # 返回完全正确预测的样本比例


def AbsoluteFalse(y_hat, y):
    """
    AbsoluteFalse 是计算错误预测的标签比例 (也称为 Hamming Loss)。

    参数:
    y_hat: 预测标签矩阵，shape为(n, m)
    y: 真实标签矩阵，shape与y_hat相同

    返回:
    absolute_false: Hamming Loss，即错误预测标签的比例
    """

    n, m = y_hat.shape

    score_k = 0
    for v in range(n):
        union = 0
        intersection = 0
        for h in range(m):
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        score_k += (union - intersection) / m  # 计算错误预测标签的比例
    return score_k / n  # 返回所有样本的平均 Hamming Loss


def evaluate(score_label, y, threshold=0.6):
    """
    评估模型的函数，计算 Aiming、Coverage、Accuracy、AbsoluteTrue、AbsoluteFalse 五个评价指标。

    参数:
    score_label: 预测分数矩阵，shape为(n, m)
    y: 真实标签矩阵，shape为(n, m)
    threshold: 阈值，用于将分数矩阵转换为二值标签矩阵

    返回:
    aiming: 精确率
    coverage: 召回率
    accuracy: 准确率
    absolute_true: 完全正确预测的比例
    absolute_false: 错误预测标签的比例 (Hamming Loss)
    """

    # 如果没有提供阈值，则使用默认阈值列表
    if threshold is None:
        threshold = [0.8, 0.5, 0.7, 0.6, 0.3, 0.3, 0.5, 0.7, 0.8, 0.7, 0.2, 0.9, 0.7, 0.8, 0.5, 0.7, 0.6, 0.8, 0.7, 0.7, 0.9]

    y_hat = score_label
    
    # 将预测分数矩阵转换为二值标签矩阵
    for i in range(len(y_hat)):
        for j in range(len(y_hat[i])):
            if y_hat[i][j] < threshold:  # 如果预测分数小于阈值，则预测标签为0
                y_hat[i][j] = 0
            else:  # 否则预测标签为1
                y_hat[i][j] = 1

    # 评估模型的五个评价指标
    aiming = Aiming(y_hat, y)
    coverage = Coverage(y_hat, y)
    accuracy = Accuracy(y_hat, y)
    absolute_true = AbsoluteTrue(y_hat, y)
    absolute_false = AbsoluteFalse(y_hat, y)

    return aiming, coverage, accuracy, absolute_true, absolute_false  # 返回五个评价指标的结果
