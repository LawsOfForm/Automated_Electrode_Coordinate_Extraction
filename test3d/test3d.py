import torch
import torch.nn as nn
from torchvision import models
from build_net import encoder, decoder, forward_encoder, forward_decoder


class Unet(nn.Module):
    def __init__(self, n_class: int = 2):
        super().__init__()

        self.e11, self.e12, self.pool1 = encoder(in_chan=4, out_chan=64)
        self.e21, self.e22, self.pool2 = encoder(in_chan=64, out_chan=128)
        self.e31, self.e32, self.pool3 = encoder(in_chan=128, out_chan=256)
        self.e41, self.e42, self.pool4 = encoder(in_chan=256, out_chan=512)
        self.e51, self.e52 = encoder(in_chan=512, out_chan=1024, has_pool=False)

        self.upconv1, self.d11, self.d12 = decoder(in_chan=1024, out_chan=512)
        self.upconv2, self.d21, self.d22 = decoder(in_chan=512, out_chan=256)
        self.upconv3, self.d31, self.d32 = decoder(in_chan=256, out_chan=128)
        self.upconv4, self.d41, self.d42 = decoder(in_chan=128, out_chan=64)

        self.outconv = nn.Conv3d(64, out_channels=n_class, kernel_size=1)

    def forward(self, x):

        xe12, xp1 = forward_encoder(x, self.e11, self.e12, self.pool1)
        xe22, xp2 = forward_encoder(xp1, self.e21, self.e22, self.pool2)
        xe32, xp3 = forward_encoder(xp2, self.e31, self.e32, self.pool3)
        xe42, xp4 = forward_encoder(xp3, self.e41, self.e42, self.pool4)
        xe52 = forward_encoder(xp4, self.e51, self.e52)

        xd12 = forward_decoder(xe52, xe42, self.upconv1, self.d11, self.d12)
        xd22 = forward_decoder(xd12, xe32, self.upconv2, self.d21, self.d22)
        xd32 = forward_decoder(xd22, xe22, self.upconv3, self.d31, self.d32)
        xd42 = forward_decoder(xd32, xe12, self.upconv4, self.d41, self.d42)

        out = self.outconv(xd42)

        return out
