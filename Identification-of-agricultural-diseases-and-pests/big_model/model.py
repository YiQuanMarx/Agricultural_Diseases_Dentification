import torch
from torch import nn
from torchvision import models
from efficientnet_pytorch import EfficientNet
from mobilenet.mobilenet_v2 import mobilenet_v2
from efficientnet  import EfficientNet

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        #self.model_1 = models.mobilenet_v2(pretrained=True)
        self.model_1 = mobilenet_v2(pretrained=True)
        #self.model_2 = EfficientNet.from_name('efficientnet-b3')
        self.model_2 = EfficientNet.from_name('efficientnet-b3')      # 1536
        self.weight = torch.load('/home/ll/python_daizuo/病虫害识别—细腻度/Bilinear_CNN-master/model/efficientnet-b3-5fb5a3c3.pth')
        self.state = { k:v for k ,v in self.weight.items() if k in self.model_2.state_dict().keys()}
        excepttion,unexception =  self.model_2.load_state_dict(self.state,strict=False)
        print("exception:",excepttion)
        print("unexception:",unexception)
        #self.model_2.load_state_dict(torch.load('./efficientnet-b7-dcc49843.pth'))         # 2560

        self.feature_1 = self.model_1.features          # 获取特征提取层
        self.feature_2 = self.model_2.extract_features  # 获取特征提取层
        self.avgpool = nn.AdaptiveAvgPool2d(1)          # 平均池化
        self.dropout = nn.Dropout(0.2)                  # 随机失活

        self.fc = nn.Linear(in_features=2816,out_features=7)        # b3 : 2816  b7: 2560 + 1280 = 3840

    def forward(self,input):
        # 将图像进行对半分
        x_1 = input[:,:,:224,:]
        x_2 = input[:,:,224:,:]
        x_3 = input

        # print(x_1.shape)
        # print(x_2.shape)

        out_1 = self.feature_1(x_1)     # 分支1
        out_2 = self.feature_1(x_2)     # 分支2
        out_3 = self.feature_2(input)   # 主分支

        # print(out_1.shape)
        # print(out_2.shape)
        # print(out_3.shape)

        cat_out = torch.cat((out_1,out_2),2)        # 拼接两个分支的H通道
        out = torch.cat((cat_out,out_3),1)          # 进行特征融合
        out = self.avgpool(out)
        out = out.flatten(start_dim=1)
        out = self.dropout(out)
        out = self.fc(out)                          # 全连接层

        return out


if __name__ == '__main__':
    model = Model()
    x = torch.randn(1,3,256,256)
    out = model(x)
    print(out.shape)
