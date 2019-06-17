# FCN_Pytorch_Simple_Implementation_FCN实现语义分割.

FCN的一个pytorch简单复现，数据集很小，是一些随机背景上的一些包的图片（所有数据集大小一共不到80M）  
关于此数据集详细信息，见文件bag_data和bag_data_mask。  
根据论文实现了FCN32s、FCN16s、FCN8s和FCNs  
使用visdom可视化，运行了100个epoch后的可视化如下图：  

![image](https://github.com/fuyongXu/FCN_Pytorch_Simple/blob/master/images/acc.png)
![image](https://github.com/fuyongXu/FCN_Pytorch_Simple/blob/master/images/test_iter_loss.svg)  
![image](https://github.com/fuyongXu/FCN_Pytorch_Simple/blob/master/images/train.png)
![image](https://github.com/fuyongXu/FCN_Pytorch_Simple/blob/master/images/test_prediction.png)

>1.1 我的运行环境  
- Windows 10
- pytorch == 1.0
- torchvision == 0.2.1
- visdom == 0.1.8.5
- OpenCV-Python == 3.4.1
>具体操作
1. 打开终端，输入  
```python train.py```
2.若没有问题可以打开浏览器输入http://localhost:8097/ 来使用visdom可视化

## 数据集
在training data和ground-truth分别有600张图片（0.jpg ~ 599.jpg）。
## 可视化
- train prediction：训练时模型的输出
- label：ground-truth
- test prediction：预测时模型的输出（每次训练都会预测，但预测数据不参与训练与backprop）
- train iter loss：训练时每一批（batch）的loss情况
- test iter loss：测试时每一批（batch）的loss情况
## 代码
1. train.py
- 训练网络与可视化
- 主函数
2. FCN.py
- FCN32s、FCN16s、FCN8s、FCNs网络定义
- VGGNet网络定义、VGG不同种类网络参数、构建VGG网络的函数
3. BagData.py
- 定义方便PyTorch读取数据的Dataset和DataLoader
- 定义数据的变换transform
4. onehot.py
- 图片的onehot编码  

如果您有任何建议或问题，欢迎随时与我联系，如果您发现了任何bug，或者您想做出贡献，请创建一个PR。:smile:
