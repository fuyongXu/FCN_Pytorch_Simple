import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,random_split
from torchvision import transforms

import cv2
from onehot import onehot

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])

class BagDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        return len(os.listdir('D:/Project/FCN_Pytorch_Simple/bag_data'))

    def __getitem__(self, idx):
        img_name = os.listdir('D:/Project/FCN_Pytorch_Simple/bag_data')[idx]
        imgA = cv2.imread('D:/Project/FCN_Pytorch_Simple/bag_data/'+img_name)
        imgA = cv2.resize(imgA,(160,160))
        imgB = cv2.imread('D:/Project/FCN_Pytorch_Simple/bag_data_mask/'+img_name)
        imgB = cv2.resize(imgB,(160,160))
        imgB = imgB/255             #img/255.0,img/127.5 - 1 第一种是对图像进行归一化，范围为[0, 1]，第二种也是对图像进行归一化，范围为[-1, 1]https://blog.csdn.net/u011276025/article/details/76050377
        imgB = imgB.astype('uint8')  #uint8是无符号八位整型，表示范围是[0, 255]的整数
        imgB = imgB.transpose(2,0,1)        #此处则是将表示通道数的2转到第一个维度上
        imgB = torch.FloatTensor(imgB)
        #print(imgB.shape)
        if self.transform:
            imgA = self.transform(imgA)

        return imgA,imgB

bag = BagDataset(transform)

train_size = int(0.9*len(bag))
test_size = len(bag) - train_size
train_dataset,test_dataset = random_split(bag,[train_size,test_size])

train_dataloader = DataLoader(train_dataset,batch_size=4,shuffle=True,num_workers=4)
test_dataloader = DataLoader(test_dataset,batch_size=4,shuffle=True,num_workers=4)

if __name__ == '__main__':
    for train_batch in train_dataloader:
        print(train_batch)

    for test_batch in test_dataloader:
        print(test_batch)