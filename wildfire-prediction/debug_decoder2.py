import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet_unet import ResNetEncoder, DoubleConv

def debug_decoder_step_by_step():
    print("Debugging decoder step by step with actual shapes...")
    
    # Get encoder outputs
    encoder = ResNetEncoder(12)
    x = torch.randn(2, 12, 64, 64)
    with torch.no_grad():
        e1, e2, e3, e4 = encoder(x)
    
    print("Encoder outputs:")
    print(f"  e1: {e1.shape}")  # [2, 64, 16, 16]
    print(f"  e2: {e2.shape}")  # [2, 128, 8, 8]
    print(f"  e3: {e3.shape}")  # [2, 256, 4, 4]
    print(f"  e4: {e4.shape}")  # [2, 512, 2, 2]
    
    # Test DecoderBlock as defined
    print("\\nTesting DecoderBlock(512, 256, 256):")
    dec4 = nn.ConvTranspose2d(512, 512, 2, stride=2)
    up4 = dec4(e4)
    print(f"  up4: {up4.shape}")
    
    # This is what the DecoderBlock does:
    # self.up = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
    # self.conv = DoubleConv(in_channels + skip_channels, out_channels)
    # So for DecoderBlock(512, 256, 256):
    # up: ConvTranspose2d(512, 512, 2, stride=2)
    # conv: DoubleConv(512 + 256, 256) = DoubleConv(768, 256)
    
    # Let's manually do what DecoderBlock.forward does:
    print("\\nManual DecoderBlock forward:")
    x = e4  # [2, 512, 2, 2]
    skip = e3  # [2, 256, 4, 4]
    
    # x = self.up(x)
    up = nn.ConvTranspose2d(512, 512, 2, stride=2)
    x = up(x)
    print(f"  after up: {x.shape}")  # Should be [2, 512, 4, 4]
    
    # if x.shape[2:] != skip.shape[2:]:
    #     x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
    if x.shape[2:] != skip.shape[2:]:
        print(f"  Size mismatch: {x.shape[2:]} vs {skip.shape[2:]}, interpolating")
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        print(f"  after interp: {x.shape}")
    else:
        print(f"  Size match: {x.shape[2:]}")
    
    # x = torch.cat([x, skip], dim=1)
    x = torch.cat([x, skip], dim=1)
    print(f"  after cat: {x.shape}")  # Should be [2, 512+256=768, 4, 4]
    
    # return self.conv(x)
    conv = DoubleConv(512 + 256, 256)  # DoubleConv(768, 256)
    x = conv(x)
    print(f"  after conv: {x.shape}")   # Should be [2, 256, 4, 4]
    
    print("\\nThis should work. Let's test the actual DecoderBlock class:")
    from models.resnet_unet import DecoderBlock
    dec4_actual = DecoderBlock(512, 256, 256)
    with torch.no_grad():
        result = dec4_actual(e4, e3)
    print(f"Actual DecoderBlock result: {result.shape}")

if __name__ == "__main__":
    debug_decoder_step_by_step()