import torch.nn as nn
from PIL import Image
import numpy as np

def encoder_block():
        return nn.Sequential(nn.Conv2d(16, 16, 3, stride=1, padding=1),
#                              nn.BatchNorm2d(16),
                             nn.ReLU(True),
                             nn.Conv2d(16, 16, 3, stride=1, padding=1),
#                              nn.BatchNorm2d(16),
                             nn.ReLU(True))
def encoder_reduce(x):
        return nn.Sequential(nn.Conv2d(16, x, 3, stride=2, padding=1),
#                              nn.BatchNorm2d(x),
                             nn.ReLU(True))

def decoder_block():
        return nn.Sequential(
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
#             nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
#             nn.BatchNorm2d(16),
            nn.ReLU(True))

def decoder_expand():
        return nn.Sequential(nn.ConvTranspose2d(16,16, 3, stride=2,padding=1, output_padding=1),
#               nn.BatchNorm2d(16),
              nn.ReLU(True))

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_start = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),  # b, 16, 10, 10
#             nn.BatchNorm2d(16),
            nn.ReLU(True))
        self.reduce_0 = encoder_reduce(16)
        self.encoder_1 = encoder_block()
        self.reduce_1 = encoder_reduce(16)
        self.encoder_2 = encoder_block()
        self.reduce_2 = encoder_reduce(16)
        self.encoder_3 = encoder_block()
        self.reduce_3 = encoder_reduce(8)

    def forward(self, x):
        out = self.encoder_start(x)
        out1 = self.reduce_0(out)
        out = self.encoder_1(out1)
        out1 = out + out1
        out1 = self.reduce_1(out1)
        out = self.encoder_2(out1)
        out1 = out + out1
        out1 = self.reduce_2(out1)
        out = self.encoder_3(out1)
        out1 = out +out1
        out1 = self.reduce_3(out1)

        return out1

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder_start = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2,padding=1, output_padding=1),  # b, 16, 5, 5
#             nn.BatchNorm2d(16),
            nn.ReLU(True))
        self.decoder_1 = decoder_block()
        self.expand_1 = decoder_expand()
        self.decoder_2 = decoder_block()
        self.expand_2 = decoder_expand()
        self.decoder_3 = decoder_block()
        self.expand_3 = decoder_expand()
        self.decoder_end = nn.Sequential(nn.Conv2d(16, 3, 3, stride=1, padding=1),
                            nn.ReLU(True))

    def forward(self, x):
        out1 = self.decoder_start(x)
        out = self.decoder_1(out1)
        out = out1 + out
        out1 = self.expand_1(out)
        out = self.decoder_2(out1)
        out = out1 + out
        out1 = self.expand_2(out)
        out = self.decoder_3(out1)
        out = out1 + out
        out1 = self.expand_3(out)
        return self.decoder_end(out1)
