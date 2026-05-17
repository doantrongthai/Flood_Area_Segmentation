import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    def __init__(self, dim, mixer_kernel=(5, 5), growth=8):
        super().__init__()
        k = mixer_kernel[0]
        self.dw1  = nn.Conv2d(dim,        growth, k, padding='same', groups=1, bias=False)
        self.bn1  = nn.BatchNorm2d(growth)
        self.act1 = nn.PReLU(growth)
        self.dw2  = nn.Conv2d(dim+growth, growth, k, padding='same', groups=1, bias=False)
        self.bn2  = nn.BatchNorm2d(growth)
        self.act2 = nn.PReLU(growth)
        self.pw   = nn.Conv2d(dim+growth*2, dim, 1, bias=False)
        self.bn3  = nn.BatchNorm2d(dim)
        self.act3 = nn.PReLU(dim)

    def forward(self, x):
        x1  = self.act1(self.bn1(self.dw1(x)))
        x2  = self.act2(self.bn2(self.dw2(torch.cat([x, x1], dim=1))))
        out = self.act3(self.bn3(self.pw(torch.cat([x, x1, x2], dim=1))))
        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(5, 5)):
        super().__init__()
        self.same_channels = (in_c == out_c)
        conv_out = out_c - in_c if not self.same_channels else out_c
        self.pfcu      = DenseBlock(in_c, mixer_kernel=mixer_kernel)
        self.bn        = nn.BatchNorm2d(in_c)
        self.down_pool = nn.MaxPool2d((2, 2))
        if not self.same_channels:
            self.pw      = nn.Conv2d(in_c, conv_out, 1, bias=False)
            self.down_pw = nn.MaxPool2d((2, 2))
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c)

    def forward(self, x):
        skip = self.bn(self.pfcu(x))
        pool = self.down_pool(skip)
        if self.same_channels:
            x = self.act(self.bn2(pool))
        else:
            conv = self.down_pw(self.pw(skip))
            x    = self.act(self.bn2(torch.cat([pool, conv], dim=1)))
        return x, skip


class BottleNeck(nn.Module):
    def __init__(self, dim, max_dim=128):
        super().__init__()
        self.dw  = nn.Conv2d(dim, dim, 5, padding='same', groups=dim, bias=False)
        self.bn  = nn.BatchNorm2d(dim)
        self.act = nn.PReLU(dim)

    def forward(self, x):
        return self.act(self.bn(x + self.dw(x)))


class Decoder(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(5, 5)):
        super().__init__()
        self.up        = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.reduce_up = nn.Conv2d(in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()
        self.bn        = nn.BatchNorm2d(out_c)
        self.act       = nn.PReLU(out_c)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.reduce_up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.act(self.bn(x + skip))


class AblModel_DenseNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        mk = (5, 5)
        self.conv_in = nn.Conv2d(3, 16, 3, padding=1)
        self.e1 = EncoderBlock(16,  32,  mixer_kernel=mk)
        self.e2 = EncoderBlock(32,  64,  mixer_kernel=mk)
        self.e3 = EncoderBlock(64,  128, mixer_kernel=mk)
        self.e4 = EncoderBlock(128, 256, mixer_kernel=mk)
        self.b4 = BottleNeck(256)
        self.d4 = Decoder(256, 128, mixer_kernel=mk)
        self.d3 = Decoder(128, 64,  mixer_kernel=mk)
        self.d2 = Decoder(64,  32,  mixer_kernel=mk)
        self.d1 = Decoder(32,  16,  mixer_kernel=mk)
        self.conv_out = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        x = self.conv_in(x)
        x, s1 = self.e1(x)
        x, s2 = self.e2(x)
        x, s3 = self.e3(x)
        x, s4 = self.e4(x)
        x = self.b4(x)
        x = self.d4(x, s4)
        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)
        return self.conv_out(x)


def build_model(num_classes=1):
    return AblModel_DenseNet(num_classes=num_classes)
