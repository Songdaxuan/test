# -*- coding:utf-8 -*-

import numpy as np
import operator

"""
创建数据集
"""


def createDataSet():
    group = np.array([[1, 10], [5, 89], [108, 5], [115, 8]])
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return group, labels


"""
KNN算法
"""


def classify0(inX, dataSet, labels, k):
    # shape返回对应数组的行列数元组（行数，列数）
    dataSetSize = dataSet.shape[0]
    # 得到各点xy轴距离
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 计算各点欧式距离
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # 返回各值大小对应的索引值
    sortedDistIndices = distances.argsort()

    classCount = {}
    for i in range(k):
        # votellabel = labels[sortedDistIndices[i]]
        minindex = np.where(sortedDistIndices == i)
        votellabel = labels[int(minindex[0])]
        classCount[votellabel] = classCount.get(votellabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


if __name__ == '__main__':
    group, labels = createDataSet()
    test = [101, 20]
    test_class = classify0(test, group, labels, 3)
    print(test_class)
