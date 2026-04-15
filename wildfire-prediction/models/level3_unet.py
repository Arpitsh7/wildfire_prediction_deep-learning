import torch
import torch.nn as nn
import torch.nn.functional as F


class LightConv(nn.Module):
    """Lightweight double convolution block."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        se = self.fc(torch.nn.functional.adaptive_avg_pool2d(x, 1).view(b, c))
        return x * se.view(b, c, 1, 1)


class Level3UNet(nn.Module):
    """
    Level 3: Compact U-Net with SE blocks and improved training strategy.
    Much faster than attention U-Net but still powerful.
    """
    def __init__(self, in_channels=12, out_channels=1):
        super().__init__()
        
        # Encoder
        self.enc1 = LightConv(in_channels, 32)
        self.se1 = SEBlock(32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = LightConv(32, 64)
        self.se2 = SEBlock(64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = LightConv(64, 128)
        self.se3 = SEBlock(128)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = LightConv(128, 256)
        self.se4 = SEBlock(256)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = LightConv(256, 512)
        self.se_bottleneck = SEBlock(512)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = LightConv(512, 256)
        self.se_dec4 = SEBlock(256)
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = LightConv(256, 128)
        self.se_dec3 = SEBlock(128)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = LightConv(128, 64)
        self.se_dec2 = SEBlock(64)
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = LightConv(64, 32)
        self.se_dec1 = SEBlock(32)
        
        self.final = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        # Encoder with SE blocks
        e1 = self.enc1(x)
        e1 = self.se1(e1)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        e2 = self.se2(e2)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        e3 = self.se3(e3)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        e4 = self.se4(e4)
        p4 = self.pool4(e4)
        
        # Bottleneck
        bottleneck = self.bottleneck(p4)
        bottleneck = self.se_bottleneck(bottleneck)
        
        # Decoder with skip connections and SE blocks
        d4 = self.upconv4(bottleneck)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d4 = self.se_dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d3 = self.se_dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d2 = self.se_dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        d1 = self.se_dec1(d1)
        
        return self.final(d1)
