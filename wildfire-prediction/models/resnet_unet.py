import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class DoubleConv(nn.Module):
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


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels=12):
        super().__init__()
        # Load pretrained ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Replace first conv layer to accept our number of input channels
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # Use pretrained layers
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        
        return e1, e2, e3, e4


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        # Handle potential size mismatch due to odd dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNetUNet(nn.Module):
    def __init__(self, in_channels=12, out_channels=1):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels)
        
        # Decoder with skip connections
        # From bottleneck (after encoder): we have e4 at 512 channels
        self.dec4 = DecoderBlock(512, 256, 256)  # e4 -> e3: 512 up + 256 skip = 768 -> 256 out
        self.dec3 = DecoderBlock(256, 128, 128)  # e3 -> e2: 256 up + 128 skip = 384 -> 128 out
        self.dec2 = DecoderBlock(128, 64, 64)    # e2 -> e1: 128 up + 64 skip = 192 -> 64 out
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 2, stride=2),  # 64 -> 64, 16x16 -> 32x32
            DoubleConv(64, 32),                       # 64 -> 32
            nn.Conv2d(32, out_channels, 1)            # final 1x1 conv to get 1 channel
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 32x32 -> 64x64
    
    def forward(self, x):
        # Encoder - get feature maps at different scales
        e1, e2, e3, e4 = self.encoder(x)
        
        # Decoder with skip connections
        d4 = self.dec4(e4, e3)  # 512 + 256 -> 256
        d3 = self.dec3(d4, e2)  # 256 + 128 -> 128
        d2 = self.dec2(d3, e1)  # 128 + 64 -> 64
        d1 = self.dec1(d2)      # 64 -> 32 -> 1
        d1 = self.upsample(d1)  # 32x32 -> 64x64
        
        return d1