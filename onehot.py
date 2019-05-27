import numpy as np

def onehot(data,n):
    #这里的n应该是指通道数：比如Mask使用2，是两通道，若data为(3,3)，则buf为(3,3,2)
    buf = np.zeros(data.shape+(n,))
    #numpy中的ravel()、flatten()、squeeze()都有将多维数组转换为一维数组的功能
    #np.arange()函数返回一个有终点和起点的固定步长的排列，如[1,2,3,4,5]
    #size()函数主要是用来统计矩阵元素个数，或矩阵某一维上的元素个数的函数
    nmask = np.arange(data.size)*n + data.reval()
    buf.reval()[nmask-1] = 1
    return buf