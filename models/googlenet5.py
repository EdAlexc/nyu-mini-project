'''GoogLeNet with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=5, padding=2),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet5(nn.Module):
    def __init__(self):
        super(GoogLeNet5, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=5, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
        )

        self.a3 = Inception(96,  32,  48, 64, 8, 16, 16) 
        self.b3 = Inception(128, 64, 64, 96, 16, 48, 32) 

        self.maxpool = nn.MaxPool2d(5, stride=2, padding=1) #11

        self.a4 = Inception(240, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 80, 56, 112, 12,  32,  32)
        self.c4 = Inception(256, 64, 64, 128, 12,  32,  32)
        self.d4 = Inception(256, 56, 72, 144, 16,  32,  32)
        self.e4 = Inception(264, 128, 80, 160, 16, 64, 64) #

        self.a5 = Inception(416, 128, 80, 160, 16, 64, 64)
        self.b5 = Inception(416, 192, 96, 192, 24, 64, 64) #

        self.avgpool = nn.AvgPool2d(7, stride=2, padding=1)
        self.linear = nn.Linear(512, 10) #21-22

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = GoogLeNet5()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

# test()
