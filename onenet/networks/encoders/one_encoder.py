import torch.nn as nn

from onenet.utils.layers import (
    Conv1DBlock,
    PixelUnshuffle,
    channel_recover,
    spatial_flatten,
)


class Bottleneck(nn.Module):
    """Chain of Conv1D for channel expansion"""

    def __init__(self, repeat=3, in_channels=3, out_channels=32):
        super(Bottleneck, self).__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1DBlock(
                    kernel_size=in_channels if i == 0 else 1,
                    in_channels=1 if i == 0 else out_channels,
                    out_channels=out_channels,
                    stride=in_channels if i == 0 else 1,
                    use_pad=False,
                )
                for i in range(repeat)
            ]
        )

    def forward(self, x):
        x, (b, c, h, w) = spatial_flatten(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        x = channel_recover(x, h, w)
        return x


class OneEncoder(nn.Module):
    def __init__(self, opts):
        """PUD > Conv1D(k) > Conv1D(C,C/2) > Conv1D(C/2,C/2)"""
        super(OneEncoder, self).__init__()

        self.opts = opts

        # Calculate channel outs
        self.channels = [(2**i) * self.opts["bottleneck_channel"] for i in range(self.opts["num_layers"] + 1)]

        # Initialize Layers
        self.bottleneck = Bottleneck(
            repeat=self.opts["bottleneck_repeat"],
            in_channels=self.opts["channel_in"],
            out_channels=self.opts["bottleneck_channel"],
        )
        self.convs = nn.ModuleDict()
        for i in range(self.opts["num_layers"]):
            self.convs[f"spatial_{i}"] = Conv1DBlock(kernel_size=self.opts["kernel_size"])
            self.convs[f"channel_1_{i}"] = Conv1DBlock(
                kernel_size=2 * self.channels[i + 1],
                out_channels=self.channels[i + 1],
                stride=2 * self.channels[i + 1],
                use_pad=False,
            )
            self.convs[f"channel_2_{i}"] = Conv1DBlock(
                kernel_size=1,
                in_channels=self.channels[i + 1],
                out_channels=self.channels[i + 1],
                stride=1,
                use_pad=False,
            )
            # self.convs[f"channel_3_{i}"] = Conv1DBlock(
            #     kernel_size=1,
            #     in_channels=self.channels[i + 1],
            #     out_channels=self.channels[i + 1],
            #     stride=1,
            #     use_pad=False,
            # )
        self.downsample = PixelUnshuffle(scale=2)

    def get_channels(self):
        """Returns encoded channels for skip connections"""
        return self.channels

    def forward(self, x):
        self.features = []

        x = (x - 0.45) / 0.225
        x = self.bottleneck(x)
        self.features.append(x)
        for i in range(self.opts["num_layers"]):
            x = self.downsample(x)
            x, (_, _, h, w) = spatial_flatten(x)
            x = self.convs[f"spatial_{i}"](x)
            x = self.convs[f"channel_1_{i}"](x)
            x = self.convs[f"channel_2_{i}"](x)
            # x = self.convs[f"channel_3_{i}"](x)
            x = channel_recover(x, h, w)

            self.features.append(x)

        return self.features
