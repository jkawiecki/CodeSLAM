###################################################################
### Code modified from https://github.com/milesial/Pytorch-UNet ###
###################################################################
import torch.nn as nn

from .blocks import DoubleConv, Down, Up, OutConv

class UNet(nn.Module):
    def __init__(self, cfg, bilinear=True):
        super(UNet, self).__init__()
        # resnet

        self.down1 = Down(3,   16)  # (3, 16)
        self.down2 = Down(16,  32)  # (16, 32)
        self.down3 = Down(32,  64)  # (32, 64)
        self.down4 = Down(64, 128)  # (64, 128)
        self.down5 = Down(128,256)  # (128, 256)

        self.d_inc = DoubleConv(256, 256)   # (256, 256)
        self.up1 =   Up(256,    128, bilinear) # (256, 128)
        self.up2 =   Up(128,    64, bilinear)  # (128, 64)
        self.up3 =   Up(64,     32, bilinear)   # (64, 32)
        self.up4 =   Up(32,     16, bilinear)   # (32, 16)
        self.out =   nn.Sequential(
            OutConv(16, 1),
            nn.Sigmoid()
        )
        


    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x1_out = self.d_inc(x5)
        x2_out = self.up1(x1_out, x4)
        x3_out = self.up2(x2_out, x3)
        x4_out = self.up3(x3_out, x2)
        x5_out = self.up4(x4_out, x1)


        out = self.out(x5_out)

        feature_maps = (x1_out, x2_out, x3_out, x4_out, x5_out)

        return  feature_maps, out