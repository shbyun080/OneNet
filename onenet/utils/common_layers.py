import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, channel_in, channel_out, repeat=3, kernel=3):
        super(Bottleneck, self).__init__()

        self.blocks = nn.ModuleList(
            [ConvBlock(channel_in if i == 0 else channel_out, channel_out, kernel=kernel) for i in range(repeat)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU"""

    def __init__(self, in_channels, out_channels, kernel=3, bn_first=True, use_relu=True):
        super(ConvBlock, self).__init__()

        self.bn_first = bn_first
        self.use_relu = use_relu

        self.conv = Conv3x3(in_channels, out_channels, kernel=kernel)
        if self.use_relu:
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            if self.bn_first:
                x = self.bn(x)
            x = self.relu(x)
            if not self.bn_first:
                x = self.bn(x)
        return x


class Conv3x3(nn.Module):
    """Layer to pad and convolve input"""

    def __init__(self, in_channels, out_channels, kernel=3, use_refl=False):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=kernel)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x


class UpSample(nn.Module):
    """Upsample with ConvTranspose2D"""

    def __init__(self, in_channels, out_channels, scale=2):
        super(UpSample, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale, stride=scale)

    def forward(self, x):
        x = self.conv(x)
        return x


def upsample(x):
    """Upsample input tensor by a factor of 2"""
    return F.interpolate(x, scale_factor=2, mode="nearest")
