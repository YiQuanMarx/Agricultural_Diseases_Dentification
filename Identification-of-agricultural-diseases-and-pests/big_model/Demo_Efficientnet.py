from efficientnet_pytorch import EfficientNet
import torch
import torchvision
from efficientnet import EfficientNet
from torchvision import models

class BCNN(torch.nn.Module):

    def __init__(self):
        super(BCNN, self).__init__()
        #self.model = EfficientNet.from_name('efficientnet-b1')       # 初始化模型
        self.model = models.mobilenet_v2(pretrained=True)
        self.feature = self.model.features                      # 获取特征层
        self.fc = torch.nn.Linear(in_features=1280**2,out_features=11,bias=True)

        print(self.feature)


    def forward(self, X):
        """Forward pass of the network.
        Args:
            X, torch.autograd.Variable of shape N*3*448*448.
        Returns:
            Score, torch.autograd.Variable of shape N*200.
        """
        N = X.size()[0]         # 获取batch_size

        assert X.size() == (N, 3, 224, 224)     # 224, 224
        X = self.feature(X)
        assert X.size() == (N, 1280, 7, 7)  #
        X = X.view(N, 1280, 7 ** 2)  # [b,320,14,14] ---> [b,320,14*14]
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (7 ** 2)  # Bilinear  transpose(X, 1, 2) ---> [b,14*14,320]
        assert X.size() == (N, 1280, 1280)
        X = X.view(N, 1280 ** 2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)            # 造成X的值为NAN ,默认对行进行二范数计算
        X = self.fc(X)
        assert X.size() == (N, 11)
        return X


class Efficient_BCNN(torch.nn.Module):
    def __init__(self):
        super(Efficient_BCNN, self).__init__()
        self.model = EfficientNet.from_name('efficientnet-b0')
        weight = torch.load('./efficientnet-b0-355c32eb.pth')
        weight_info = {k: v for k, v in weight.items() if k in self.model.state_dict().items()}
        missing_keys, unexpected_keys = self.model.load_state_dict(weight_info, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)
        self.feature = self.model
        self.fc = torch.nn.Linear(in_features=320**2,out_features=11,bias=True)

    def forward(self,X):
        N = X.size()[0]  # 获取batch_size
        assert X.size() == (N, 3, 224, 224)  # 224, 224
        X = self.feature(X)
        assert X.size() == (N, 320, 7, 7)  #
        X = X.view(N, 320, 7 ** 2)  # [b,320,14,14] ---> [b,320,14*14]
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (7 ** 2)  # Bilinear  transpose(X, 1, 2) ---> [b,14*14,320]
        assert X.size() == (N, 320, 320)
        X = X.view(N, 320 ** 2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)  # 造成X的值为NAN ,默认对行进行二范数计算
        X = self.fc(X)
        assert X.size() == (N, 11)
        return X



if __name__ == '__main__':
    modle = BCNN()


