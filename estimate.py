#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 10:28
# @Author  : ywh
# @File    : estimate.py
# @Software: PyCharm

# 多标签评价指标
def Aiming(y_hat, y):
    """
    the “Aiming” rate (also called “Precision”) is to reflect the average ratio of the
    correctly predicted labels over the predicted labels; to measure the percentage
    of the predicted labels that hit the target of the real labels.
    """

    n, m = y_hat.shape  # n表示肽的数量， m表示肽的功能数量

    score_k = 0
    for v in range(n): # 遍历每个样本
        union = 0
        intersection = 0
        for h in range(m): # 遍历每个样本的功能标签
            if y_hat[v, h] == 1 or y[v, h] == 1:  # L ∪ L*  计算并集
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:  # L ∩ L* ，计算交集
                intersection += 1
        if intersection == 0:
            continue
        score_k += intersection / sum(y_hat[v]) # 累计
    return score_k / n #取平均


def Coverage(y_hat, y):
    """
    The “Coverage” rate (also called “Recall”) is to reflect the average ratio of the
    correctly predicted labels over the real labels; to measure the percentage of the
    real labels that are covered by the hits of prediction.
    """

    n, m = y_hat.shape # n表示样本的数量，m表示每个样本的功能数量（标签数量）

    score_k = 0 # 初始化
    for v in range(n): # 遍历每个样本
        union = 0
        intersection = 0
        for h in range(m): # 遍历每个样本的功能标签
            if y_hat[v, h] == 1 or y[v, h] == 1:
                union += 1
            if y_hat[v, h] == 1 and y[v, h] == 1:
                intersection += 1
        if intersection == 0:
            continue
        score_k += intersection / sum(y[v])

    return score_k / n


def Accuracy(y_hat, y):
    """
    The “Accuracy” rate is to reflect the average ratio of correctly predicted labels
    over the total labels including correctly and incorrectly predicted labels as well
    as those real labels but are missed in the prediction
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
        score_k += intersection / union
    return score_k / n


def AbsoluteTrue(y_hat, y):
    """
    same
    """

    n, m = y_hat.shape
    score_k = 0
    for v in range(n):
        if list(y_hat[v]) == list(y[v]):
            score_k += 1
    return score_k / n



def AbsoluteFalse(y_hat, y):
    """
    hamming loss
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
        score_k += (union - intersection) / m
    return score_k / n


def evaluate(score_label, y, threshold=0.6):
    # 将预测概率分数转为标签
    if threshold is None:
        threshold = [0.8, 0.5, 0.7, 0.6, 0.3, 0.3, 0.5, 0.7, 0.8, 0.7, 0.2, 0.9, 0.7, 0.8, 0.5, 0.7, 0.6, 0.8, 0.7, 0.7, 0.9]

    y_hat = score_label
    for i in range(len(y_hat)):
        for j in range(len(y_hat[i])):
            if y_hat[i][j] < threshold:  # threshold
                y_hat[i][j] = 0
            else:
                y_hat[i][j] = 1

    # 评估模型
    aiming = Aiming(y_hat, y)
    coverage = Coverage(y_hat, y)
    accuracy = Accuracy(y_hat, y)
    absolute_true = AbsoluteTrue(y_hat, y)
    absolute_false = AbsoluteFalse(y_hat, y)


    return aiming, coverage, accuracy, absolute_true, absolute_false



