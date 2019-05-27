import numpy as np

data =np.array([[1,2,3],[4,5,6],[7,8,9]])
buf = np.zeros(data.shape+(2,))
    #numpy中的ravel()、flatten()、squeeze()都有将多维数组转换为一维数组的功能
nmask = np.arange(data.size)*2
print(data.shape)
print(buf)
print(buf.shape)
print(nmask)