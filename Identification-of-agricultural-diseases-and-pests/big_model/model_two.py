import torch
from torch import nn
from torchvision import models
from efficientnet_pytorch import EfficientNet


class Model_two(nn.Module):
    def __init__(self):
        super(Model_two, self).__init__()
        self.model_1 = models.mobilenet_v2(pretrained=True)
        self.model_2 = EfficientNet.from_name('efficientnet-b7')
        self.model_2.load_state_dict(torch.load('./efficientnet-b7-dcc49843.pth'))
        self.feature_1 = self.model_1.features
        self.feature_2 = self.model_2.extract_features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)

        self.fc = nn.Linear(in_features=3840,out_features=4)

    def forward(self,input):
        x_1 = input
        x_2 = input

        out_1 = self.feature_1(x_1)
        out_2 = self.feature_2(x_2)

        out = torch.cat((out_1,out_2),1)
        out = self.avgpool(out)
        out = out.flatten(start_dim=1)
        out = self.dropout(out)
        out = self.fc(out)

        return out

if __name__ == '__main__':
    model = Model_two()
    x = torch.randn(1,3,256,256)
    out = model(x)
    print(out.shape)
