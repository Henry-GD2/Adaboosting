#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2020/6/30 15:46


"""
1.实现单层决策树,找出数据集上最佳的单层决策树
2.实现多层的分类（adaboost）
"""

import numpy as np
import matplotlib.pyplot as plt


# 加载文件
def loadDataSet(path = 'horseColicTraining2.txt'):
    data = list()
    labels = list()
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip().split(',')
            lineArr = []
            for i in range(len(line)-1):
                lineArr.append(float(line[i]))
            data.append(lineArr)
            labels.append(float(line[-1]))
        xMat = np.array(data)
        yMat = np.array(labels).reshape(-1, 1)
    return xMat, yMat


# 按照阈值分类结果
def Classify0(xMat, i, Q, S):
    """
    xMat:数据矩阵
    Q：阈值
    S：标志
    """
    re = np.ones((xMat.shape[0], 1))
    if S == 'lt':
        re[xMat[:, i] <= Q] = -1  # 如果小于阈值，则赋值为1
    else:
        re[xMat[:, i] > Q] = 1  # 如果大于阈值，则赋值为1
    return re


# 找出数据集上最佳的单层决策树
def get_Stump(xMat, yMat, D):
    """
    参数说明：
        xMat:特征矩阵
        yMat:标签矩阵
        D：样本权重
    返回：
        bestStump:最佳单层决策树信息
        minE:最小误差
        bestClass:最佳的分类结果
    """
    m, n = xMat.shape  #m为样本的个数，n为特征数
    Step = 10  # 初始化一个步数
    bestStump = {}  # 用字典形式来存储树桩信息
    bestClass = np.mat(np.zeros((m, 1)))  # 初始化分类结果为1
    minE = np.inf  # 最小误差初始化为正无穷大
    for i in range(n):  # 遍历所有特征值
        min = xMat[:, i].min()  # 找到特征的最小值
        max = xMat[:, i].max()  # 找到特征的最大值
        stepSize = (max - min)/ Step  # 计算步长
        for j in range(-1, int(Step)+1):
            for S in ['lt', 'gt']:  # 大于和小于的情况，均遍历
                Q = (min + j * stepSize)  # 计算阈值
                re = Classify0(xMat, i, Q, S)  # 计算分类结果
                err = np.mat(np.ones((m,1)))  # 初始化误差矩阵
                err[re == yMat] = 0  # 分类正确的，赋值为0
                eca = D.T * err  # 计算误差
                if eca < minE:  # 找到误差最小的分类方式
                    minE = eca
                    bestClass = re.copy()
                    bestStump['特征值'] = i
                    bestStump['阈值'] = Q
                    bestStump['标志'] = S
    return bestStump, minE, bestClass


# 基于单层决策树的Adaboost训练过程
def Ada_train(xMat, yMat, maxC=40):
    """
    函数功能：基于单层决策树的Adaboost训练过程
    参数说明:
        xMat:特征矩阵
        yMat:标签矩阵
        maxC:最大迭代次数
    返回：
        weakClass:弱分类器信息
        aggClass:类别估值（更改标签估值）
    """
    weakClass = []
    m = xMat.shape[0]
    D = np.mat(np.ones((m, 1))/m)  # 初始化权重
    aggClass = np.mat(np.zeros((m,1)))
    for i in range(maxC):
        Stump, error, bestClass = get_Stump(xMat, yMat, D)  # 构造单层决策树
        alpha = float(0.5*np.log((1-error)/max(error, 1e-6)))  # 计算弱分类器权重alpha
        Stump['alpha'] = np.round(alpha, 2)
        weakClass.append(Stump)  # 存储单层决策树
        expon = np.multiply(-1*alpha*yMat, bestClass)  # 计算e的指数项
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()  # 更新权重
        aggClass += alpha * bestClass  # 计算类别估值（更改标签估值）
        aggErr = np.multiply(np.sign(aggClass) != yMat, np.ones((m, 1)))  # 统计误分类样本数
        errRate = aggErr.sum()/m
        print("total error: ", errRate)
        if errRate == 0:
            break
    return weakClass, aggClass


# 开始对待预测的数据进行分类
def AdaClassify(xMat, weakClass):
    m = xMat.shape[0]  # 待分类数据集的长度
    aggClass = np.mat(np.zeros((m, 1)))

    for i in range(len(weakClass)):  # 遍历所有分类器进行分类
        classEst = Classify0(xMat,
                             weakClass[i]['特征值'],
                             weakClass[i]['阈值'],
                             weakClass[i]['标志'],
                             )
        aggClass += classEst * weakClass[i]['alpha']
    return np.sign(aggClass)


# ROC图像
def plotROC(predStrengths, classLabels):
    """
    输入：
        predStrengths : Adaboost预测的结果（行）
        classLabels : 原本训练数据的标签（列）
    """
    cur = (1.0, 1.0)  # 光标
    ySum = 0.0  # 变量来计算AUC
    numPosClas = sum(np.array(classLabels) == 1.0)
    yStep = 1 / float(numPosClas)  # 向上的步长
    xStep = 1 / float(len(classLabels) - numPosClas)  # 向右的步长
    sortedIndicies = predStrengths.argsort()  # 得到排序索引，它是反向的
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # 循环所有的值，在每个点上画一条线段
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        # draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')  # 逐渐加入曲线变化的一条直线
        cur = (cur[0] - delX, cur[1] - delY)  # 重新更新cur的起始点

    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("the Area Under the Curve is: ", ySum * xStep)


if __name__ == '__main__':
    xMat, yMat = loadDataSet(path='horseColicTraining2.txt')  # 训练数据

    # 测试单层决策树
    m = xMat.shape[0]
    D = np.mat(np.ones((m, 1))/m)
    bestStump, minE, bestClass = get_Stump(xMat, yMat, D)

    # 测试单层决策树的Adaboost训练过程
    weakClass, aggClass = Ada_train(xMat, yMat, maxC=10)  # 返回弱分类器的集合，以及弱分类的标签值
    print('分类器的个数:', len(weakClass))
    testArr, testLabelArr = loadDataSet(path='horseColicTest2.txt')  # 测试数据
    pre = AdaClassify(testArr, weakClass)  # 返回预测值

    # 计算准确度
    errArr = np.mat(np.ones((len(pre), 1)))  # 一共有m个预测样本
    cnt = errArr[pre != testLabelArr].sum()
    print('误分类点在总体预测样本中的比例为：', cnt / len(pre))
    print(weakClass)
    print('lt代表：如果小于等于阈值，则赋值为1 \n gt代表：如果大于阈值，则赋值为1')
    # 绘画出ROC图
    plotROC(aggClass.T, yMat)