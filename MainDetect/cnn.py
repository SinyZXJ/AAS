import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F

'''开始建立CNN网络'''
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        '''
        一般来说，卷积网络包括以下内容：
        1.卷积层
        2.神经网络
        3.池化层
        '''
        self.conv1=nn.Sequential(
            nn.Conv2d(              #--> (3,100,100)
                in_channels=3,      #传入的图片是几层的，灰色为1层，RGB为三层
                out_channels=5,    #输出的图片是几层
                kernel_size=5,      #代表扫描的区域点为5*5
                stride=1,           #就是每隔多少步跳一下
                padding=2,          #边框补全，其计算公式=（kernel_size-1）/2=(5-1)/2=2
            ),    # 2d代表二维卷积           -->    (16,100,100)
            nn.ReLU(),              #非线性激活层
            nn.MaxPool2d(kernel_size=2),    #设定这里的扫描区域为2*2，且取出该2*2中的最大值          --> (16,50,50)
        )
 
        self.conv2=nn.Sequential(
            nn.Conv2d(              #       -->  (16,50,50)
                in_channels=16,     #这里的输入是上层的输出为16层
                out_channels=1,    #在这里我们需要将其输出为32层
                kernel_size=5,      #代表扫描的区域点为5*5
                stride=1,           #就是每隔多少步跳一下
                padding=2,          #边框补全，其计算公式=（kernel_size-1）/2=(5-1)/2=
            ),                      #   --> (32,50,50)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),    #设定这里的扫描区域为2*2，且取出该2*2中的最大值     --> 这里是三维数据(32,25,25)
        )

        self.fc1 = nn.Linear(25*25, 25*25)  # 隐藏层
        self.fc2 = nn.Linear(12500, 20)  # 隐藏层
        self.fc3 = nn.Linear(20,1)  # 输出层
        self.out=nn.Linear(1,1)       #注意一下这里的数据是二维的数据
        #self.ad=nn.ReLU()
        
    def forward(self,x):
        x=F.relu(self.conv1(x))
        self.weights =self.conv1[0].weight.data
        #print(weights)
        #x=F.relu(self.conv2(x))     #（batch,32,7,7）
        #然后接下来进行一下扩展展平的操作，将三维数据转为二维的数据
        x=x.view(x.size(0),-1)    #(batch ,32 * 7 * 7)
        #x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        output=self.out(x)
        return output