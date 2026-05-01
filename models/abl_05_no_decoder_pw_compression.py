import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        k = mixer_kernel[0]
        self.dw = nn.Conv2d(dim, dim, kernel_size=k, padding='same', groups=dim, dilation=dilation, bias=False)

    def forward(self, x):
        return x + self.dw(x)


class Axial_PFCU_Single(nn.Module):
    def __init__(self, dim, mixer_kernel=(5, 5)):
        super().__init__()
        self.branch_r1 = StandardDW(dim, mixer_kernel, dilation=1)
        self.pw_fuse   = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.bn_fuse   = nn.BatchNorm2d(dim)
        self.act       = nn.PReLU(dim)

    def forward(self, x):
        b1 = self.branch_r1(x)
        fused = self.bn_fuse(self.pw_fuse(b1))
        return self.act(fused + x)


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(5, 5)):
        super().__init__()
        self.same_channels = (in_c == out_c)
        conv_out = out_c - in_c if not self.same_channels else out_c
        self.pfcu      = Axial_PFCU_Single(in_c, mixer_kernel=mixer_kernel)
        self.bn        = nn.BatchNorm2d(in_c)
        self.down_pool = nn.MaxPool2d((2, 2))
        if not self.same_channels:
            self.pw      = nn.Conv2d(in_c, conv_out, kernel_size=1, bias=False)
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


class SimpleBottleNeck(nn.Module):
    def __init__(self, dim, max_dim=128):
        super().__init__()
        self.dw  = StandardDW(dim, mixer_kernel=(5, 5), dilation=1)
        self.bn  = nn.BatchNorm2d(dim)
        self.act = nn.PReLU(dim)

    def forward(self, x):
        return self.act(self.bn(self.dw(x)))


class DecoderBlock_NoPWCompression(nn.Module):
    def __init__(self, in_c, out_c, mixer_kernel=(5, 5)):
        super().__init__()
        self.up        = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.reduce_up = nn.Conv2d(in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()
        self.pfcu      = Axial_PFCU_Single(out_c, mixer_kernel=mixer_kernel)
        self.bn        = nn.BatchNorm2d(out_c)
        self.act       = nn.PReLU(out_c)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.reduce_up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = x + skip
        x = self.act(self.bn(self.pfcu(x) + x))
        return x


class AblModel_NoDecoderPWCompression(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        mk = (5, 5)
        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.e1 = EncoderBlock(16,  32,  mixer_kernel=mk)
        self.e2 = EncoderBlock(32,  64,  mixer_kernel=mk)
        self.e3 = EncoderBlock(64,  128, mixer_kernel=mk)
        self.e4 = EncoderBlock(128, 256, mixer_kernel=mk)
        self.b4 = SimpleBottleNeck(256, max_dim=128)
        self.d4 = DecoderBlock_NoPWCompression(256, 128, mixer_kernel=mk)
        self.d3 = DecoderBlock_NoPWCompression(128, 64,  mixer_kernel=mk)
        self.d2 = DecoderBlock_NoPWCompression(64,  32,  mixer_kernel=mk)
        self.d1 = DecoderBlock_NoPWCompression(32,  16,  mixer_kernel=mk)
        self.conv_out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)
        x = self.b4(x)
        x = self.d4(x, skip4)
        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)
        return self.conv_out(x)


def build_model(num_classes=1):
    return AblModel_NoDecoderPWCompression(num_classes=num_classes)
