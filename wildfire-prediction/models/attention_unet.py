import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        # Calculate padding to maintain spatial dimensions with dilation
        padding = dilation
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return self.sigmoid(out).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # Channel attention
        x = x * self.channel_attention(x)
        # Spatial attention
        x = x * self.spatial_attention(x)
        return x


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=12, out_channels=1):
        super().__init__()
        
        # Encoder with Dilated/Atrous Convolutions
        self.enc1 = DoubleConv(in_channels, 64, dilation=1)
        self.enc2 = DoubleConv(64, 128, dilation=2)
        self.enc3 = DoubleConv(128, 256, dilation=4)
        self.enc4 = DoubleConv(256, 512, dilation=8)
        
        # Channel Attention Mechanism (CBAM) for encoder features
        self.ca1 = CBAM(64)
        self.ca2 = CBAM(128)
        self.ca3 = CBAM(256)
        self.ca4 = CBAM(512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024, dilation=1)
        self.ca_bottleneck = CBAM(1024)
        
        # Decoder with Attention Gates and Feature Recalibration
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.attg4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.dec4 = DoubleConv(1024, 512)
        self.ca_dec4 = CBAM(512)  # Feature recalibration for decoder stage 4
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.attg3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec3 = DoubleConv(512, 256)
        self.ca_dec3 = CBAM(256)  # Feature recalibration for decoder stage 3
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.attg2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec2 = DoubleConv(256, 128)
        self.ca_dec2 = CBAM(128)  # Feature recalibration for decoder stage 2
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.attg1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec1 = DoubleConv(128, 64)
        self.ca_dec1 = CBAM(64)   # Feature recalibration for decoder stage 1
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder with Multi-Scale Feature Pyramid and Channel Attention
        enc1 = self.enc1(x)
        enc1 = self.ca1(enc1)
        enc2 = self.enc2(self.pool(enc1))
        enc2 = self.ca2(enc2)
        enc3 = self.enc3(self.pool(enc2))
        enc3 = self.ca3(enc3)
        enc4 = self.enc4(self.pool(enc3))
        enc4 = self.ca4(enc4)
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        bottleneck = self.ca_bottleneck(bottleneck)
        
        # Decoder with Attention Gates and Feature Recalibration
        dec4 = self.upconv4(bottleneck)
        att4 = self.attg4(g=dec4, x=enc4)
        concat4 = torch.cat((att4, enc4), dim=1)
        dec4 = self.dec4(concat4)
        dec4 = self.ca_dec4(dec4)  # Feature recalibration for decoder stage 4
        
        dec3 = self.upconv3(dec4)
        att3 = self.attg3(g=dec3, x=enc3)
        concat3 = torch.cat((att3, enc3), dim=1)
        dec3 = self.dec3(concat3)
        dec3 = self.ca_dec3(dec3)  # Feature recalibration for decoder stage 3
        
        dec2 = self.upconv2(dec3)
        att2 = self.attg2(g=dec2, x=enc2)
        concat2 = torch.cat((att2, enc2), dim=1)
        dec2 = self.dec2(concat2)
        dec2 = self.ca_dec2(dec2)  # Feature recalibration for decoder stage 2
        
        dec1 = self.upconv1(dec2)
        att1 = self.attg1(g=dec1, x=enc1)
        concat1 = torch.cat((att1, enc1), dim=1)
        dec1 = self.dec1(concat1)
        dec1 = self.ca_dec1(dec1)  # Feature recalibration for decoder stage 1
        
        return self.final(dec1)