import torch
import torchvision

class BCNN(torch.nn.Module):
    """B-CNN for CUB200.
    The B-CNN model is illustrated as follows.
    conv1^2 (64) -> pool1 -> conv2^2 (128) -> pool2 -> conv3^3 (256) -> pool3
    -> conv4^3 (512) -> pool4 -> conv5^3 (512) -> bilinear pooling
    -> sqrt-normalize -> L2-normalize -> fc (200).
    The network accepts a 3*448*448 input, and the pool5 activation has shape
    512*28*28 since we down-sample 5 times.
    Attributes:
        features, torch.nn.Module: Convolution and pooling layers.
        fc, torch.nn.Module: 200.
    """
    def __init__(self):
        """Declare all needed layers."""
        torch.nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.features = torchvision.models.vgg16(pretrained=False).features
        print(self.features)
        self.features = torch.nn.Sequential(*list(self.features.children())
                                            [:-1])  # Remove 最后的Maxpool2d层
        print(self.features)
        # Linear classifier.
        self.fc = torch.nn.Linear(512**2, 11)

    def forward(self, X):
        """Forward pass of the network.
        Args:
            X, torch.autograd.Variable of shape N*3*448*448.
        Returns:
            Score, torch.autograd.Variable of shape N*200.
        """
        N = X.size()[0]         # 获取batch_size
        assert X.size() == (N, 3, 224, 224)
        X = self.features(X)
        assert X.size() == (N, 512, 14, 14)         #
        X = X.view(N, 512, 14**2)                   # [b,512,28,28] ---> [b,512,28*28]
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (14**2)  # Bilinear  transpose(X, 1, 2) ---> [b,28*28,512]
        assert X.size() == (N, 512, 512)
        X = X.view(N, 512**2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, 11)
        return X

if __name__ == '__main__':
    model = BCNN()
    x = torch.randn(1,3,224,224)
    out = model(x)
    print(out)
    print(out.shape)