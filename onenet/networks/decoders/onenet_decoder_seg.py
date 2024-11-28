import torch
import torch.nn as nn

from onenet.utils.layers import Conv1DBlock, PixelShuffle, channel_recover, spatial_flatten


class OneDecoder(nn.Module):
    def __init__(self, opts, channel_enc):
        super(OneDecoder, self).__init__()

        self.opts = opts
        self.channel_enc = channel_enc
        self.len_ch_enc = len(self.channel_enc)

        # Initialize Layers
        self.bottleneck = Conv1DBlock(
            kernel_size=self.channel_enc[-1],
            out_channels=2 * self.channel_enc[-1],
            stride=self.channel_enc[-1],
            use_pad=False,
        )

        self.convs = nn.ModuleDict()
        for i in range(self.len_ch_enc - 1, 0, -1):
            self.convs[f"spatial_{i}"] = Conv1DBlock(kernel_size=self.opts["kernel_size"])
            self.convs[f"channel_1_{i}"] = Conv1DBlock(
                kernel_size=2 * self.channel_enc[i - 1] if i > 1 else self.channel_enc[i - 1],
                out_channels=2 * self.channel_enc[i - 1] if i > 1 else self.channel_enc[i - 1],
                stride=2 * self.channel_enc[i - 1] if i > 1 else self.channel_enc[i - 1],
                use_pad=False,
            )
            self.convs[f"channel_2_{i}"] = Conv1DBlock(
                kernel_size=1,
                in_channels=2 * self.channel_enc[i - 1] if i > 1 else self.channel_enc[i - 1],
                out_channels=2 * self.channel_enc[i - 1] if i > 1 else self.channel_enc[i - 1],
                stride=1,
                use_pad=False,
            )
            self.convs[f"head_{i}"] = Conv1DBlock(
                kernel_size=1,
                in_channels=2 * self.channel_enc[i - 1] if i > 1 else self.channel_enc[i - 1],
                out_channels=self.opts["channel_out"],
                stride=1,
                use_relu=False,
                use_pad=False,
                use_bn=False,
            )

        self.upsample = PixelShuffle(scale=2)

    def forward(self, features_enc):
        self.outputs = []

        x = features_enc[-1]
        x, (_, _, h, w) = spatial_flatten(x)
        x = self.bottleneck(x)
        x = channel_recover(x, h, w)
        for i in range(self.len_ch_enc - 1, 0, -1):
            x = [self.upsample(x)]
            if i > 1:  # play with 0/1
                x += [features_enc[i - 1]]
            x = torch.cat(x, 1)
            x, (_, _, h, w) = spatial_flatten(x)
            x = self.convs[f"spatial_{i}"](x)
            x = self.convs[f"channel_1_{i}"](x)
            x = self.convs[f"channel_2_{i}"](x)
            if i == 1:
                self.outputs.append(channel_recover(self.convs[f"head_{i}"](x), h, w))
            x = channel_recover(x, h, w)

        return self.outputs
