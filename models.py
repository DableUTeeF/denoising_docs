from torch import nn
from timm.models.convnext import convnextv2_base
import torch
from torch.nn import functional as F


class Upsampling(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 ):
        super().__init__()
        self.transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.transpose(x)
        x = self.bn(x)
        return self.act(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = convnextv2_base(pretrained=True)
        self.upsampling = nn.ModuleList([
            Upsampling(1024, 512, 2),
            Upsampling(512, 256, 2),
            Upsampling(256, 128, 2),
            Upsampling(128, 64, 2),
            Upsampling(64, 32, 2),
        ])
        self.output = nn.Conv2d(32, 3, 1)

    def forward(self, xx):
        x0 = self.backbone.stem(xx)
        x1 = self.backbone.stages[0](x0)
        x2 = self.backbone.stages[1](x1)
        x3 = self.backbone.stages[2](x2)
        x = self.backbone.stages[3](x3)
        x = self.backbone.norm_pre(x)

        x = F.interpolate(self.upsampling[0](x), x3.size()[2:]) + x3
        del x3
        x = F.interpolate(self.upsampling[1](x), x2.size()[2:]) + x2
        del x2
        x = F.interpolate(self.upsampling[2](x), x1.size()[2:]) + x1
        del x1
        x = F.interpolate(self.upsampling[3](x), x0.size()[2:])
        del x0
        x = F.interpolate(self.upsampling[4](x), xx.size()[2:])
        x = self.output(x)
        return x


if __name__ == '__main__':
    model = Model()
    x = torch.ones((1, 3, 64, 256))
    model(x)
