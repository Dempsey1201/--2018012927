import numpy as np


# 计算样本的信息熵(经验熵),该部分为自己自己参考源码写出的
def calcEntropy(data):
    """
    输入：
        data: array
            样本数据，data数据中的最后一列代表样本的类别
    返回:
        float
            样本的信息熵
    """
    sample_size = data.shape[0]
    label_list = list(data[:, -1])

    entropy = 0.0
    for label in set(label_list):
        prob = float(label_list.count(label) / sample_size)
        entropy -= prob * np.log2(prob)

    return entropy
