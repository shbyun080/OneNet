import torch
import torch.nn as nn


def spatial_flatten(x):
    """Flattens to Bx1xHWC and returns tensor with original b,c,h,w values"""
    ratio = x.size()
    x = torch.permute(x, (0, 2, 3, 1))
    x = x.reshape(x.size()[0], 1, -1)
    return x, ratio


def spatial_recover(x, ratio):
    """Recovers flattened shape of Bx1xHWC to given b,c,h,w ratio"""
    x = x.reshape(ratio[0], ratio[2], ratio[3], ratio[1])
    x = torch.permute(x, (0, 3, 1, 2))
    return x


def channel_flatten(x):
    """Flattens B x C x HW to Bx1xHWC"""
    x = torch.permute(x, (0, 2, 1))
    x = x.reshape(x.size()[0], 1, -1)
    return x


def channel_recover(x, h, w):
    """Recovers BxCxHW to BxCxHxW given h and w"""
    x = x.reshape(x.size()[0], x.size()[1], h, w)
    return x


# Conv1D
class Conv1DBlock(nn.Module):
    """Layer to pad and convolve input Bx1xHWC -> Bx1xHWC or BxCxHW"""

    def __init__(
        self,
        kernel_size,
        in_channels=1,
        out_channels=1,
        stride=1,
        use_refl=True,
        use_relu=True,
        use_pad=True,
        use_bn=True,
    ):
        super(Conv1DBlock, self).__init__()

        self.out_channels = out_channels
        self.use_relu = use_relu
        self.use_pad = use_pad
        self.use_bn = use_bn

        if use_refl:
            self.pad = nn.ReflectionPad1d(kernel_size // 2)
        else:
            self.pad = nn.ZeroPad1d(kernel_size // 2)
        self.conv = nn.Conv1d(in_channels, self.out_channels, kernel_size, stride=stride)
        if self.out_channels > 1 and self.use_bn:
            self.bn = nn.BatchNorm1d(self.out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.use_pad:
            x = self.pad(x)
        x = self.conv(x)
        if self.use_relu:
            x = self.relu(x)
        if self.out_channels > 1 and self.use_bn:
            x = self.bn(x)
        return x


# MaxPool
class Pool1D(nn.Module):
    """Layer to maxpool Bx1xHWC -> Bx1xHWC/2"""

    def __init__(self, kernel_size):
        super(Pool1D, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size, stride=2, padding=kernel_size // 2)

    def forward(self, x):
        x = self.pool(x)
        return x


# Upscaling, call Conv1D afterwards
class UpScale1D(nn.Module):
    """Upsamples channel dim by Bx1xHWC -> Bx1x2HWC"""

    def __init__(self, scale):
        super(UpScale1D, self).__init__()
        self.upscale = nn.Upsample(scale_factor=scale, mode="linear")

    def forward(self, x):
        x = self.upscale(x)
        return x


# PixelShuffle
class PixelShuffle(nn.Module):
    """Upsamples image by BxCxHxW -> BxC/4x2Hx2W"""

    def __init__(self, scale):
        super(PixelShuffle, self).__init__()
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.shuffle(x)
        return x


# PixelUnshuffle
class PixelUnshuffle(nn.Module):
    """Upsamples image by BxCxHxW -> BxC/4x2Hx2W"""

    def __init__(self, scale):
        super(PixelUnshuffle, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(scale)

    def forward(self, x):
        x = self.unshuffle(x)
        return x
