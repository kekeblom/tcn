import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torch.autograd import Function

class BatchNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.batch_norm(x)
        return F.relu(x, inplace=True)

class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation

    def forward(self, x):
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x, inplace=True)
        return x

class EmbeddingNet(nn.Module):
    def normalize(self, x):
        buffer = torch.pow(x, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        normalization_constant = torch.sqrt(normp)
        output = torch.div(x, normalization_constant.view(-1, 1).expand_as(x))
        return output

class PosNet(EmbeddingNet):
    def __init__(self):
        super(PosNet, self).__init__()
        # Input 1
        self.Conv2d_1a = nn.Conv2d(3, 64, bias=False, kernel_size=10, stride=2)
        self.Conv2d_2a = BatchNormConv2d(64, 32, bias=False, kernel_size=3, stride=1)
        self.Conv2d_3a = BatchNormConv2d(32, 32, bias=False, kernel_size=3, stride=1)
        self.Conv2d_4a = BatchNormConv2d(32, 32, bias=False, kernel_size=2, stride=1)

        self.Dense1 = Dense(6 * 6 * 32, 32)
        self.alpha = 10

    def forward(self, input_batch):
        # 128 x 128 x 3
        x = self.Conv2d_1a(input_batch)
        # 60 x 60 x 64
        x = self.Conv2d_2a(x)
        # 58 x 58 x 64
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # 29 x 29 x 32
        x = self.Conv2d_3a(x)
        # 27 x 27 x 32
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # 13 x 13 x 32
        x = self.Conv2d_4a(x)
        # 12 x 12 x 32
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size()[0], -1)
        # 6 x 6 x 32
        x = self.Dense1(x)
        # 32

        return self.normalize(x) * self.alpha

class TCNModel(EmbeddingNet):
    def __init__(self, inception):
        super().__init__()
        self.transform_input = True
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Conv2d_6a_3x3 = BatchNormConv2d(288, 100, kernel_size=3, stride=1)
        self.Conv2d_6b_3x3 = BatchNormConv2d(100, 20, kernel_size=3, stride=1)
        self.SpatialSoftmax = nn.Softmax2d()
        self.FullyConnected7a = Dense(31 * 31 * 20, 32)

        self.alpha = 10.0

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 33 x 33 x 100
        x = self.Conv2d_6a_3x3(x)
        # 31 x 31 x 20
        x = self.Conv2d_6b_3x3(x)
        # 31 x 31 x 20
        x = self.SpatialSoftmax(x)
        # 32
        x = self.FullyConnected7a(x.view(x.size()[0], -1))

        # Normalize output such that output lives on unit sphere.
        # Multiply by alpha as in https://arxiv.org/pdf/1703.09507.pdf
        return self.normalize(x) * self.alpha


def define_model(pretrained=True):
    return TCNModel(models.inception_v3(pretrained=pretrained))
