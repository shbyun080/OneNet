import torch
import torch.nn as nn

from onenet.utils.common_layers import Conv3x3, ConvBlock, UpSample


class CommonSegDecoder(nn.Module):
    def __init__(self, opts, channel_enc):
        super(CommonSegDecoder, self).__init__()

        self.opts = opts
        self.channel_enc = channel_enc
        self.len_ch_enc = len(self.channel_enc)

        # Initialize Layers
        self.last_layer = -1 if self.opts["name"] == "RESNET" else 0
        self.skip_stop = self.last_layer + 1 if self.opts["name"] == "RESNET" else self.last_layer + 1

        self.convs = nn.ModuleDict()
        for i in range(self.len_ch_enc - 1, self.last_layer, -1):
            self.convs[f"up_{i}"] = UpSample(self.channel_enc[i], self.channel_enc[i - 1 if i > self.skip_stop else 0])
            self.convs[f"channel_1_{i}"] = ConvBlock(
                self.channel_enc[i - 1 if i > self.skip_stop else 0] * (2 if i > self.skip_stop else 1),
                self.channel_enc[i - 1 if i > self.skip_stop else 0],
                use_relu=i > self.last_layer + 1,
            )
            self.convs[f"channel_2_{i}"] = ConvBlock(
                self.channel_enc[i - 1 if i > self.skip_stop else 0],
                self.channel_enc[i - 1 if i > self.skip_stop else 0],
                use_relu=i > self.last_layer + 1,
            )
            self.convs[f"head_{i}"] = Conv3x3(
                self.channel_enc[i - 1 if i > self.skip_stop else 0], self.opts["channel_out"]
            )

    def forward(self, features_enc):
        self.outputs = []

        x = features_enc[-1]

        for i in range(self.len_ch_enc - 1, self.last_layer, -1):
            x = [self.convs[f"up_{i}"](x)]
            if i > self.skip_stop:
                x += [features_enc[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[f"channel_1_{i}"](x)
            x = self.convs[f"channel_2_{i}"](x)
            if i == self.last_layer + 1:
                self.outputs.append(self.convs[f"head_{i}"](x))

        return self.outputs
