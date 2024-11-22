import time

import torch
import torch.nn as nn

from onenet.networks.decoders.common_decoder_seg import CommonSegDecoder
from OneNet.onenet.networks.decoders.onenet_decoder_seg import OneDecoder
from onenet.networks.encoders.one_encoder import OneEncoder


class OneNet(nn.Module):

    model_types = {
        "SEGONE": {"segmentation": OneDecoder},
        "ONENET": {"segmentation": CommonSegDecoder},
    }

    def __init__(self, opts):
        super(OneNet, self).__init__()
        self.opts = opts

        assert self.opts["name"] in self.model_types
        assert self.opts["type"] in self.model_types[self.opts["name"]]

        self.encoder = OneEncoder(self.opts)
        self.decoder = self.model_types[self.opts["name"]][self.opts["type"]](
            self.opts, channel_enc=self.encoder.get_channels()
        )

    def forward(self, x):
        enc_features = self.encoder(x)
        outputs = self.decoder(enc_features)
        return outputs
